import os
import sys
import transformers
import fire
import json
import numpy as np
import dill
import torch
import argparse
import math
import re
from data import SeqRecDataset
from my_trainer import MyGRPOTrainer
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM
from trl import GRPOConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from utils import *


def train_topk(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    ddp = world_size > 1
    if local_rank == 0:
        print(vars(args))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = 0

    model = Qwen3ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )
    model = model.merge_and_unload() 
    
    conf_tokens = ["<conf_neg>", "<conf_pos>"]

    if conf_tokens[0] not in tokenizer.all_special_tokens:
        if local_rank == 0:
            print("Adding special tokens...")
            print("Before adding, vocab size: ", len(tokenizer))
        tokenizer.add_special_tokens({'additional_special_tokens': conf_tokens})
        if local_rank == 0:
            print("After adding, vocab size: ", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

        if args.smart_initialize_conf_tokens:
            neg_anchor_id = tokenizer.encode(args.negative_word, add_special_tokens=False)
            pos_anchor_id = tokenizer.encode(args.positive_word, add_special_tokens=False)
            
            if local_rank == 0:
                print("Smartly initializing conf tokens...")
                print(f"negative_word: '{args.negative_word}' ids: {neg_anchor_id}")
                print(f"positive_word: '{args.positive_word}' ids: {pos_anchor_id}")
            
            if len(neg_anchor_id) != 1 or len(pos_anchor_id) != 1:
                raise ValueError("negative_word and positive_word should be single token words.")
            
            neg_anchor_id = neg_anchor_id[0]
            pos_anchor_id = pos_anchor_id[0]

            conf_token_ids = tokenizer.convert_tokens_to_ids(conf_tokens)

            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()

            with torch.no_grad():
                target_neg_id = conf_token_ids[0]
                input_embeddings.weight[target_neg_id].copy_(input_embeddings.weight[neg_anchor_id].clone())
                output_embeddings.weight[target_neg_id].copy_(output_embeddings.weight[neg_anchor_id].clone())

                target_pos_id = conf_token_ids[1]
                input_embeddings.weight[target_pos_id].copy_(input_embeddings.weight[pos_anchor_id].clone())
                output_embeddings.weight[target_pos_id].copy_(output_embeddings.weight[pos_anchor_id].clone())

            if local_rank == 0:
                print(f"Smart Initialization completed.")
                print(f"  <conf_neg> initialized from '{args.negative_word}'")
                print(f"  <conf_pos> initialized from '{args.positive_word}'")
                
                print("Check initialized conf token embeddings and LM head weights:")
                for i, token_str in enumerate(conf_tokens): 
                    token_id = conf_token_ids[i]
                    emb_weight = model.get_input_embeddings().weight[token_id]
                    lm_head_weight = model.get_output_embeddings().weight[token_id]
                    print(f"  {token_str} (id: {token_id}):")
                    print(f"    Embedding: {emb_weight[:5]} ...")
                    print(f"    LM Head:   {lm_head_weight[:5]} ...")

            
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)


    for n, p in model.named_parameters():
        if "original_module" in n and any(module_name in n for module_name in lora_config.modules_to_save):
            p.requires_grad = False

    if local_rank == 0:
        model.print_trainable_parameters()
        print("Trainable parameters:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"{n}: {p.shape}")

    train_data = SeqRecDataset(args, mode="train_GRPO")
    eval_data = SeqRecDataset(args, mode="eval_GRPO")

    prefix_allowed_tokens = train_data.get_prefix_allowed_tokens_fn(tokenizer)

    if local_rank == 0:
        print("Train data size: ", len(train_data))
        print("Eval data size: ", len(eval_data))
        print("Sample train data: ", train_data[0])
        print("Sample eval data: ", eval_data[0])

    def SID_acc_reward(prompts, completions, gt, **kwargs):

        raise NotImplementedError("The reward function 'SID_acc_reward' has been moved to MyGRPOTrainer class in my_grpo_trainer.py.")
    
    Trainer = MyGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        reward_funcs=[SID_acc_reward],
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        beam_search=args.beam_search,
        test_during_training=args.test_during_training,
        advantage_sum_weight=args.advantage_sum_weight,
        item_len=args.item_len,
        alpha_min=args.alpha_min,
        coeff_partial=args.coeff_partial,
        coeff_wrong=args.coeff_wrong,
        args=GRPOConfig(
            seed=args.seed,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            optim=args.optim,
            output_dir=args.output_dir,
            save_strategy=args.save_and_eval_strategy,
            eval_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            deepspeed=args.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            report_to=['tensorboard'],
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            warmup_ratio=args.warmup_ratio,
            num_generations=args.num_generations,
            beta=0,
        )   
    )

    model.config.use_cache = False
    Trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_rl_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train_topk(args)