import os
import math

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .datasets import PsyDialogueDataset
from lm.utils import SPECIAL_TOKENS, create_dataset_entry_qa, CrossEntropyLoss, PolyLoss, RMCLSLoss, RMLoss, MetricComputer, default_preprocess

model_path = {
    'chatglm': 'chatglm-6b',
    'llama': 'llama_hf',
    'baichuan': 'baichuan2-7b-base',
    'bloomz': 'bloomz_mt'
}


def get_loss(loss, poly_eps: float = 1.0, score_l2_reg: float = 0.001):
    if loss == "CrossEntropyLoss":
        return CrossEntropyLoss()
    elif loss == "Poly":
        return PolyLoss(epsilon=poly_eps)
    elif loss == "RMLoss":
        return RMLoss(beta=score_l2_reg)
    elif loss == "RMCLSLoss":
        return RMCLSLoss()
    else:
        raise ValueError(f"Loss {loss} not supported")


def get_tokenizer(args):
    model_name_or_path = os.path.join(args.cache_dir, model_path[args.model_name])
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir, use_fast=False)

    tokenizer.add_special_tokens({
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token
    })
    additional_special_tokens = (
        [] if 'additional_special_tokens' not in tokenizer.special_tokens_map else tokenizer.special_tokens_map[
            'additional_special_tokens']
    )
    additional_special_tokens = list(set(additional_special_tokens + list(SPECIAL_TOKENS.values())))
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    return tokenizer


def tokenizer_sanity_check(tokenizer):
    print('Tokenizer sanity check:')
    print(f'Type: {type(tokenizer).__name__}')

    print('special_tokens_map:', tokenizer.special_tokens_map)

    print(f'bos_token={tokenizer.bos_token}, bos_token_id={tokenizer.bos_token_id}')
    print(f'eos_token={tokenizer.eos_token}, eos_token_id={tokenizer.eos_token_id}')

    ds_entry = create_dataset_entry_qa(
        questions=['Q1', 'Q2'], answers=['A1', 'A2']
    )
    in_text = ds_entry.get_formatted(
        tokenizer.eos_token,
        use_system_tag=True,
        system_property_dropout=0,
        system_add_length=False
    )
    in_text = ''.join(in_text)

    prompter_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['Patient'])
    assistant_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['Doctor'])
    print(f'{prompter_token_id=}, {assistant_token_id=}')

    print('input text:', in_text)
    tr = tokenizer(in_text, max_length=1024, pad_to_max_length=False, truncation=True)
    print('encoding result:')
    for key, value in tr.items():
        print(f' {key}:', value)

    message_indices = []
    i = -1
    for id in tr.input_ids:
        if id in (prompter_token_id, assistant_token_id):
            i += 1
        message_indices.append(i)

    for i, xs in enumerate(tr.input_ids):
        decoded = tokenizer.decode(xs)
        print(f'{i}: {xs} -> {decoded}')

    print('message_indices:', message_indices)


def get_model(args, tokenizer, pad_vocab_size_to_multiple_of=16):
    dtype = torch.float32
    if args.dtype in ['fp16', 'float16']:
        dtype = torch.float16
    elif args.dtype in ['bf16', 'bfloat16']:
        dtype = torch.bfloat16
    model_name_or_path = os.path.join(args.cache_dir, model_path[args.model_name])
    if 'glm' in model_name_or_path:
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir, torch_dtype=dtype).half()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=args.cache_dir, torch_dtype=dtype, trust_remote_code=True,)

    num_embeddings = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != num_embeddings:
        p = pad_vocab_size_to_multiple_of
        target_size = math.ceil(len(tokenizer) / p) * p
        model.resize_token_embeddings(target_size)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in model_parameters])
    print(f'Number of trainable parameters: {int(params / 1e6)}M')

    return model


def get_dataset(args):
    train_set = PsyDialogueDataset(args.data_dir)
    eval_set = PsyDialogueDataset(args.data_dir, data_type='eval')
    return train_set, eval_set


def get_metrics(args, tokenizer):
    metrics, preprocess_fns = [MetricComputer(tokenizer)], [default_preprocess]
    return metrics, preprocess_fns
