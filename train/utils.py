import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset

def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--base_model", type=str,
                        default="./llama-7b/",
                        help="basic model path")
    parser.add_argument("--output_dir", type=str,
                        default="./ckpt/",
                        help="The output directory")


    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="the item indices file")

    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1,1,1,1,1,1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="0,0,0,100000,0,0",
                        help="the number of sampling prompts for each task")

    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=False,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=1,
                        help="the number of sampling validation sequential recommendation prompts")

    return parser

def parse_train_args(parser):

    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj", help="separate by comma")
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default="./LETTER-LC-Rec/config/ds_z2_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=1.0)

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")

    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")


    return parser

def parse_rl_args(parser):
    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path for RL training")
    parser.add_argument("--beam_search", action="store_true", default=False,
                        help="whether use beam search during GRPO training")
    parser.add_argument("--test_during_training", action="store_true", default=False,
                        help="whether test the model during GRPO training")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="the number of generations for each prompt during GRPO training")
    parser.add_argument("--advantage_sum_weight", type=float, default=0.0,
                        help="the weight for summing token rewards into advantages")
    parser.add_argument("--item_len", type=int, default=4,
                        help="the number of SIDs each item has")
    parser.add_argument("--alpha_min", type=float, default=0.1,
                        help="the minimum value of alpha for dynamic weight calculation")
    parser.add_argument("--coeff_partial", type=float, default=0.01,
                        help="the coefficient for partial item loss")
    parser.add_argument("--coeff_wrong", type=float, default=0.1,
                        help="the coefficient for wrong item loss")
    parser.add_argument("--smart_initialize_conf_tokens", action="store_true", default=False,
                        help="whether smartly initialize confidence tokens")
    parser.add_argument("--negative_word", type=str, default=" wrong",
                        help="the word indicating inconfidence")
    parser.add_argument("--positive_word", type=str, default=" correct",
                        help="the word indicating confidence")
    
    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
