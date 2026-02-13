import argparse
import json
import os
import sys

import torch
import transformers
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

from utils import *
from collator import TestCollator
from prompt import all_prompt
from evaluate import get_hit_results, calculate_global_hit, calculate_global_ndcg


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_ddp(args):

    set_seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path, padding_side="left")
    tokenizer.pad_token_id = 0

    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    if args.lora:
        if local_rank == 0:
            print("Loading base model from:", args.base_model)
        model = Qwen3ForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='cpu',
        )

        model.resize_token_embeddings(len(tokenizer))
        
        if local_rank == 0:
            print("Loading LoRA from:", args.ckpt_path)
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
        )
        model = model.merge_and_unload()

        if args.ckpt_path2 is not None:

            if local_rank == 0:
                print("Loading second LoRA from:", args.ckpt_path2)
            
            tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path2, padding_side="left")
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(
                model,
                args.ckpt_path2,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
            )
            model = model.merge_and_unload()

        model.to(device)
    else:
        model = Qwen3ForCausalLM.from_pretrained(
            args.ckpt_path,
            torch_dtype=torch.bfloat16,              
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    model = DistributedDataParallel(model, device_ids=[local_rank])

    test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, drop_last=True, shuffle=False)

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)

    prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer)


    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, num_workers=2, pin_memory=True)

    if local_rank == 0:
        print("data num:", len(test_data))
        print("example:", test_data[0])

    model.eval()

    all_hit_pos_container = []
    all_scores_container = []
    all_gt_container = []       
    all_output_container = [] 
    all_conf_score_container = []


    conf_tokens = ["<conf_neg>", "<conf_pos>"]
    conf_token_ids = tokenizer.convert_tokens_to_ids(conf_tokens)
    

    conf_start_ids = tokenizer.encode(" Confidence scores:", add_special_tokens=False)
    conf_start_ids_tensor = torch.tensor(conf_start_ids, dtype=torch.long, device=device)
    
    conf_values = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
    
    if local_rank == 0:
        print(f"Conf Token IDs: {conf_token_ids}")
        print(f"Conf Suffix IDs: {conf_start_ids}")
    
    with torch.no_grad():

        test_loader.dataset.set_prompt(0)

        for step, batch in enumerate(tqdm(test_loader)):
            inputs = batch[0].to(device)
            targets = batch[1]
            bs = len(targets)
            num_beams = args.num_beams
            while True:
                try:
                    output = model.module.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=10,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        early_stopping=True,
                        pad_token_id=0,
                    )
                    break
                except torch.cuda.OutOfMemoryError as e:
                    print("Out of memory!", flush=True)
                    num_beams = num_beams -1
                    print("Beam:", num_beams, flush=True)
                except Exception:
                    raise RuntimeError

            output_ids = output["sequences"]
            scores = output["sequences_scores"]

            output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            batch_hit_pos, batch_outputs = get_hit_results(output, targets, num_beams)

            batch_scores_tensor = scores.view(bs, num_beams)
            batch_scores_list = batch_scores_tensor.cpu().tolist()

            batch_gt_list = list(targets)

            trimmed_input_ids = output_ids[:, :-1]

            current_bs_beam = trimmed_input_ids.size(0)
            conf_suffix_batch = conf_start_ids_tensor.unsqueeze(0).expand(current_bs_beam, -1)
            
            conf_input_ids = torch.cat([trimmed_input_ids, conf_suffix_batch], dim=1)
            

            conf_attention_mask = torch.ones_like(conf_input_ids, device=device)

            conf_outputs = model(input_ids=conf_input_ids, attention_mask=conf_attention_mask)

            last_token_logits = conf_outputs.logits[:, -1, :]

            target_logits = last_token_logits[:, conf_token_ids] # shape: [bs*beam, 2]

            probs = F.softmax(target_logits, dim=-1) # shape: [bs*beam, 2]
            

            expected_conf_scores = (probs * conf_values).sum(dim=1) # shape: [bs*beam]
            

            batch_conf_scores_tensor = expected_conf_scores.view(bs, num_beams)
            batch_conf_scores_list = batch_conf_scores_tensor.cpu().tolist()

            gather_hit_pos = [None for _ in range(world_size)]
            dist.all_gather_object(obj=batch_hit_pos, object_list=gather_hit_pos)

            gather_scores = [None for _ in range(world_size)]
            dist.all_gather_object(obj=batch_scores_list, object_list=gather_scores)

            gather_gt = [None for _ in range(world_size)]
            dist.all_gather_object(obj=batch_gt_list, object_list=gather_gt)

            gather_output = [None for _ in range(world_size)]
            dist.all_gather_object(obj=batch_outputs, object_list=gather_output)

            gather_conf_scores = [None for _ in range(world_size)]
            dist.all_gather_object(obj=batch_conf_scores_list, object_list=gather_conf_scores)


            if local_rank == 0:
                for device_hits in gather_hit_pos:
                    all_hit_pos_container.extend(device_hits)
                
                for device_scores in gather_scores:
                    all_scores_container.extend(device_scores)
                    
                for device_gt in gather_gt:
                    all_gt_container.extend(device_gt)
                    
                for device_out in gather_output:
                    all_output_container.extend(device_out)

                for device_conf in gather_conf_scores:
                    all_conf_score_container.extend(device_conf)


            dist.barrier()

        metrics_results = {}
        if local_rank == 0:
            print("Gathering complete. Saving detailed log...")

            results_dir = os.path.dirname(args.results_file)
            if not os.path.exists(results_dir) and results_dir != '':
                os.makedirs(results_dir)

            detailed_log = []
            for i in range(len(all_hit_pos_container)):
                sample_log = {
                    "id": i,
                    "ground_truth": all_gt_container[i],
                    "hit_position": all_hit_pos_container[i],
                    "scores": all_scores_container[i],
                    "outputs": all_output_container[i],
                    "conf_scores": all_conf_score_container[i]
                }
                detailed_log.append(sample_log)
                
            detail_file_path = args.results_file.replace(".json", "_details.json")
            with open(detail_file_path, "w", encoding='utf-8') as f:
                json.dump(detailed_log, f, indent=4, ensure_ascii=False)
            print(f"Detailed logs saved to: {detail_file_path}")


            print("Calculating metrics...")
            metrics_results = {}
            metrics = args.metrics.split(",")

            for m in metrics:
                m = m.strip()
                try:

                    if "@" in m:
                        metric_name, k_str = m.split("@")
                        k = int(k_str)
                    else:
                        metric_name = m
                        k = args.num_beams

                    metric_lower = metric_name.lower()

                    if metric_lower.startswith("hit"):
                        metrics_results[m] = calculate_global_hit(all_hit_pos_container, k)
                    
                    elif metric_lower.startswith("ndcg"):
                        metrics_results[m] = calculate_global_ndcg(all_hit_pos_container, k)
                        
                except Exception as e:
                    print(f"Error calculating metric {m}: {e}")
                    import traceback
                    traceback.print_exc()            

            print("======================================================")
            print("Results: ", metrics_results)
            print("======================================================")

            with open(args.results_file, "w") as f:
                json.dump(metrics_results, f, indent=4)
            print("Metrics saved to: ", args.results_file)

    dist.barrier()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test_ddp(args)
