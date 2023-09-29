import copy
import os.path

import torch
from torch.utils.data import Dataset

from lm.utils import read_json


class PsyDialogueDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_type='train'):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.data = []
        self.no_loss_spans = []
        self.load_data()

    def load_data(self):
        data_path = os.path.join(self.data_dir, f'{self.data_type}.json')
        samples = read_json(data_path)
        for sample in samples:
            meta_prompt = '我想让你担任一位心理咨询师，请你运用心理学知识为患者进行问诊。'
            instruction_ids = self.tokenizer.encode(meta_prompt)
            assert isinstance(instruction_ids, list) and len(instruction_ids) > 0
            input_ids = copy.deepcopy(instruction_ids)
            no_loss_spans = [(0, len(instruction_ids))]
            for line in sample:
                cur_no_loss_spans = []
                if line['speaker'] == 'patient':
                    line = '<|patient|>' + line['content'] + self.tokenizer.eos_token
                else:
                    line = '<|doctor|>' + line['content'] + self.tokenizer.eos_token
                cur_turn_ids = self.tokenizer.encode(line)
                assert isinstance(cur_turn_ids, list) and len(cur_turn_ids) > 0
                if len(input_ids + cur_turn_ids) > 2048:
                    break
                input_ids.extend(cur_turn_ids)
                no_loss_spans.extend(cur_no_loss_spans)
            if len(input_ids) == len(instruction_ids):
                continue
            assert 0 < len(input_ids) <= 2048
            self.data.append(input_ids)
            self.no_loss_spans.append(no_loss_spans)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        no_loss_spans = copy.deepcopy(self.no_loss_spans[index])
        data = torch.LongTensor(data)
        attn_mask = torch.ones_like(data, dtype=torch.bool)
        label = copy.deepcopy(data)
        for no_loss_span in no_loss_spans:
            label[no_loss_span[0]: no_loss_span[1]] = -100
        return {'input_ids': data, 'attention_mask': attn_mask, 'labels': label}

    def __len__(self):
        return len(self.data)
