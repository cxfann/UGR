import copy
import random
import argparse
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
from prompt import sft_prompt, all_prompt
import numpy as np

class SeqRecDataset(Dataset):
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self._prefix_to_next_single = None
        self._allowed_by_pos_single = None
        self.all_items = None

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.prompts = all_prompt["seqrec"]


        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == 'train_GRPO':
            self.inter_data = self._process_train_GRPO_data()
        elif self.mode == 'eval_GRPO':
            self.inter_data = self._process_valid_GRPO_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _load_and_parse_file(self, filename):
        file_path = os.path.join(self.data_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        processed_data = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) > 0 and ("user_id" in lines[0] or "item_id" in lines[0]):
            lines = lines[1:]
            
        for line in tqdm(lines, desc=f"Loading {filename}"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                raise ValueError(f"Line format error: {line}")
            
            user_id = parts[0]
            history_str = parts[1]
            target_id = parts[2]
            
            history_ids = history_str.split(' ')
            
            try:
                history_tokens = ["".join(self.indices[str(hid)]) for hid in history_ids if str(hid) in self.indices]
                # 映射 Target ID -> Token
                if str(target_id) not in self.indices:
                    raise ValueError(f"Target ID not found in indices: {target_id}")
                target_token = "".join(self.indices[str(target_id)])
            except KeyError as e:
                raise KeyError(f"ID mapping error in line: {line}. Error: {e}")

            if self.add_prefix:
                history_tokens = [str(k+1) + ". " + item_tok for k, item_tok in enumerate(history_tokens)]
            
            one_data = dict()
            one_data["item"] = target_token
            one_data["inters"] = self.his_sep.join(history_tokens)
            
            processed_data.append(one_data)
            
        return processed_data

    def _process_train_GRPO_data(self):
        return self._load_and_parse_file(f"{self.dataset}.train.inter")
    
    def _process_valid_GRPO_data(self):
        return self._load_and_parse_file(f"{self.dataset}.valid.inter")
    
    def _process_test_data(self):
        inter_data = self._load_and_parse_file(f"{self.dataset}.test.inter")

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            real_sample_num = min(self.sample_num, len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, real_sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data    



    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        input = sft_prompt.format(instruction = instruction, response = "")

        return input, response
    
    def _get_GRPO_item(self, index):
        
        d = self.inter_data[index]

        prompt = self.prompts[0]
        instruction = prompt["instruction"].format(**d)
        response = prompt["response"].format(**d)
        input = sft_prompt.format(instruction=instruction, response="")

        return dict(prompt=input, gt=response)



    def __getitem__(self, index):
        
        if self.mode == 'train_GRPO' or self.mode == 'eval_GRPO':
            return self._get_GRPO_item(index)

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        prompt = self.prompts[0]

        input, output = self._get_text_data(d, prompt)

        return dict(input_ids=input, labels=output)
    
    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items
    
    def get_prefix_allowed_tokens_fn(self, tokenizer, item_len=4):
        if self._prefix_to_next_single is None:
            prefix_to_next = {}
            allowed_by_pos = {i: set() for i in range(item_len)}
            items = []

            for index in self.indices.values():
                if len(index) != item_len:
                    raise ValueError(f"Item length mismatch: expected {item_len}, got {len(index)} for item {index}")
                tok_ids = []
                for pos, tok in enumerate(index):
                    # 优先使用 convert_tokens_to_ids（适用于已注册的 special tokens）
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    # 若返回 None 或 unk，则 fallback 到 tokenizer(tok)
                    if tid is None or (hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id):
                        ids = tokenizer(tok)["input_ids"]
                        if len(ids) != 1:
                            raise ValueError(
                                f"Token `{tok}` tokenizes to multiple ids {ids}. Register it as a single special token."
                            )
                        tid = ids[0]
                    tok_ids.append(tid)
                    allowed_by_pos[pos].add(tid)
                items.append(tuple(tok_ids))

            for item in items:
                for prefix_len in range(0, item_len):
                    prefix = item[:prefix_len]
                    next_tok = item[prefix_len]
                    prefix_to_next.setdefault(prefix, set()).add(next_tok)

            for k, s in list(prefix_to_next.items()):
                prefix_to_next[k] = list(s)
            for pos in allowed_by_pos:
                allowed_by_pos[pos] = list(allowed_by_pos[pos])

            self._prefix_to_next_single = prefix_to_next
            self._allowed_by_pos_single = allowed_by_pos

        prefix_to_next = self._prefix_to_next_single
        allowed_by_pos = self._allowed_by_pos_single

        sep = tokenizer(" Response:")["input_ids"]
        sep_rev = sep[::-1]
        eos_id = tokenizer.eos_token_id

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sent = sentence.tolist() if hasattr(sentence, "tolist") else list(sentence)
            rev = sent[::-1]

            for offset in range(len(rev)):
                if offset + len(sep_rev) > len(rev):
                    break
                if rev[offset: offset + len(sep_rev)] == sep_rev:
                    tokens_after_sep = offset
                    if tokens_after_sep == 0:
                        cur_prefix = ()
                        return prefix_to_next.get(cur_prefix, None)

                    if tokens_after_sep >= item_len:
                        return [eos_id]

                    tokens_after = sent[-tokens_after_sep:]
                    pos_in_item = tokens_after_sep
                    cur_prefix = tuple(tokens_after[:pos_in_item])

                    allowed = prefix_to_next.get(cur_prefix, None)

                    if allowed is None:
                        return allowed_by_pos[tokens_after_sep]
                    return allowed

            raise ValueError(
                f'sep not found when sentence is {sentence}'
            )

        return prefix_allowed_tokens_fn
