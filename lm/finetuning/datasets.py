import copy
import os.path
import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase, TruncationStrategy

from lm.utils import read_json, format_system_prefix, format_pairs, DatasetEntrySft, DatasetEntryLm, SPECIAL_TOKENS


class PsyDialogueDataset(Dataset):
    def __init__(self, data_dir, data_type='train'):
        super().__init__()
        self.data_dir = data_dir
        self.data_type = data_type
        self.data = []
        self.load_data()

    def load_data(self):
        data_path = os.path.join(self.data_dir, f'{self.data_type}.json')
        samples = read_json(data_path)
        self.data = samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GLMPromptDataSet(Dataset):

    def __init__(self, data_dir, data_type='train'):
        data_path = os.path.join(data_dir, f'glm_{data_type}.json')
        self.all_data = read_json(data_path)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        instance = self.all_data[index]
        return instance


@dataclass
class DialogueDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 mix_length_threshold: Optional[int] = 256,
                 mix_probability: Optional[float] = 0.6,
                 pad_to_multiple_of: Optional[int] = None,
                 samples_mixing: Optional[bool] = False,
                 random_offset_probability: Optional[float] = 0.5,
                 label_masking: bool = True, use_system_prefix: bool = False, system_prefix: str = None,
                 use_system_tag: bool = False,
                 system_property_dropout: float = 0.5,
                 system_add_length: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.mix_length_threshold = mix_length_threshold
        self.mix_probability = mix_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.samples_mixing = samples_mixing
        self.random_offset_probability = random_offset_probability
        self.label_masking = label_masking
        self.use_system_prefix = use_system_prefix
        self.system_prefix = system_prefix
        self.use_system_tag = use_system_tag
        self.system_property_dropout = system_property_dropout
        self.system_add_length = system_add_length
        self.__post_init__()

    def __post_init__(self):
        assert self.tokenizer.eos_token
        if self.use_system_prefix:
            assert self.system_prefix
            self.system_prefix = self.tokenizer.encode(
                format_system_prefix(self.system_prefix, self.tokenizer.eos_token),
                add_special_tokens=False,
                return_tensors='np'
            )[0]
            self.max_length = self.max_length - len(self.system_prefix)

    def process_one(self, messages, return_length=False):
        total_short_context_one = 0
        if random.random() < self.random_offset_probability and not isinstance(messages, DatasetEntryLm):
            truncation = TruncationStrategy.DO_NOT_TRUNCATE
            max_length = None
        else:
            truncation = TruncationStrategy.LONGEST_FIRST
            max_length = self.max_length

        pretrain_dataset = False
        if isinstance(messages, DatasetEntrySft):
            messages = messages.get_formatted(
                eos_token=self.tokenizer.eos_token,
                use_system_tag=self.use_system_tag,
                system_property_dropout=self.system_property_dropout,
                system_add_length=self.system_add_length,
            )
        elif isinstance(messages, DatasetEntryLm):
            messages = messages.text
            pretrain_dataset = True
        else:
            messages = list(messages)
            messages = format_pairs(messages, self.tokenizer.eos_token)

        flatten_message = self.tokenizer(
            "".join(messages),
            max_length=max_length,
            truncation=truncation,
            padding=False
        )

        if pretrain_dataset:
            label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)
            return flatten_message, label_mask, 0

        if return_length:
            return min(len(flatten_message.input_ids), self.max_length)

        message_indices: Optional[list[int]] = None
        if self.label_masking:
            prompter_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["Patient"])
            assistant_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["Doctor"])
            assert prompter_token_id >= 0 and assistant_token_id >= 0

            message_indices = []
            i = -1
            for x in flatten_message.input_ids:
                if x in (prompter_token_id, assistant_token_id):
                    i += 1
                message_indices.append(i)

        input_length = len(flatten_message.input_ids)
        if self.max_length and input_length > self.max_length:
            offset = random.randint(0, input_length - self.max_length)
            for k in flatten_message.keys():
                v = flatten_message[k]
                if isinstance(v, list) and len(v) == input_length:
                    flatten_message[k] = v[offset: offset + self.max_length]
            if message_indices:
                message_indices = message_indices[offset: offset + self.max_length]

        if self.label_masking:
            label_mask = np.array(list(map(lambda x: x % 2 == 1, message_indices)))
        else:
            label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)

        label_mask[-1] = False  # make sure last token is inactive, has an effect only when truncating

        if len(flatten_message.input_ids) < self.mix_length_threshold and self.samples_mixing:
            total_short_context_one += len(flatten_message.input_ids)

        return {k: v for k, v in flatten_message.items() if k != "offset_mapping"}, label_mask, total_short_context_one

    def __call__(self, features):
        flatten_messages = []
        label_masks = []
        total_short_context = 0
        for messages in features:
            flatten_message, label_mask, total_short_context_one = self.process_one(messages)
            flatten_messages.append(flatten_message)
            label_masks.append(label_mask)
            total_short_context += total_short_context_one

        # packing
        if total_short_context > 2 and self.samples_mixing:
            _flatten_messages, _label_masks = [], []
            prev_short_msg, prev_short_mask = None, None
            for flatten_msg, label_mask in zip(flatten_messages, label_masks):
                if len(flatten_msg.input_ids) < self.mix_length_threshold and random.random() > self.mix_probability:
                    if prev_short_msg is not None:
                        for key in flatten_msg.keys():
                            flatten_msg[key] += prev_short_msg[key]
                            flatten_msg[key] = flatten_msg[key][: self.max_length]
                        label_mask = np.concatenate([label_mask, prev_short_mask])
                        _label_masks.append(label_mask[: self.max_length])
                        _flatten_messages.append(flatten_msg)
                        # reset
                        prev_short_msg, prev_short_mask = None, None
                    else:
                        # prime
                        prev_short_msg, prev_short_mask = flatten_msg, label_mask
                else:
                    _label_masks.append(label_mask)
                    _flatten_messages.append(flatten_msg)
            if prev_short_msg is not None:
                for key in flatten_msg.keys():
                    flatten_msg[key] += prev_short_msg[key]
                    flatten_msg[key] = flatten_msg[key][: self.max_length]
                label_mask = np.concatenate([label_mask, prev_short_mask])[: self.max_length]
                _label_masks.append(label_mask)
                _flatten_messages.append(flatten_msg)

            label_masks = _label_masks
            flatten_messages = _flatten_messages

        if self.use_system_prefix:
            flatten_messages = [
                {
                    "input_ids": np.concatenate([self.system_prefix, flatten_msg["input_ids"]]),
                    "attention_mask": np.concatenate(
                        [np.ones_like(self.system_prefix).astype(bool), flatten_msg["attention_mask"]]
                    ),
                }
                for flatten_msg in flatten_messages
            ]
            label_masks = [
                np.concatenate([np.zeros_like(self.system_prefix).astype(bool), label_mask])
                for label_mask in label_masks
            ]

        if 'glm' in type(self.tokenizer).__name__.lower():
            flatten_messages = [{'input_ids': instance['input_ids']} for instance in flatten_messages]

        batch = self.tokenizer.pad(
            flatten_messages,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        dim = batch.input_ids.shape[-1]

        batch["label_masks"] = torch.stack(
            [F.pad(torch.tensor(x), (0, dim - len(x)), value=False) for x in label_masks]
        )
        batch["targets"] = torch.roll(batch.input_ids, -1, -1)

        return batch


@dataclass
class GLMDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length, max_src_length, system_prefix, use_system_prefix):
        self.tokenizer = tokenizer
        self.system_prefix = system_prefix
        self.use_system_prefix = use_system_prefix
        self.max_length = max_length
        self.max_src_length = max_src_length

    def __call__(self, batch):
        new_batch = []
        for sample in batch:
            inputs, target = sample[:-1], sample[-1]
            messages = ''.join(format_pairs(inputs, self.tokenizer.eos_token, add_initial_reply_token=True))
            if self.use_system_prefix:
                instruction = format_system_prefix(self.system_prefix, self.tokenizer.eos_token)
                messages = instruction + messages
            src_tokens = self.tokenizer.tokenize(messages)
            if len(src_tokens) > self.max_src_length:
                src_tokens = src_tokens[-self.max_src_length:]

            max_tgt_len = self.max_length - len(src_tokens) - 3
            tgt_tokens = self.tokenizer.tokenize(target['content'])

            if len(tgt_tokens) > max_tgt_len:
                tgt_tokens = tgt_tokens[:max_tgt_len]

            tokens = src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            context_length = input_ids.index(self.tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            assert len(input_ids) == len(labels)
            assert len(input_ids) <= self.max_length

            new_batch.append({'input_ids': input_ids, 'target': labels})

        lengths = [len(instance["input_ids"]) for instance in new_batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in new_batch:
            input_ids = instance["input_ids"]
            labels = instance["target"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
        target = torch.tensor(labels_batch, dtype=torch.long)
        label_masks = -100 * torch.ones_like(target)
        label_masks = ~target.eq(label_masks)

        return {
            "input_ids": input_ids,
            "target": target,
            'label_masks': label_masks
        }
