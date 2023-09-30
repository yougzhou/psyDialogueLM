import os

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .datasets import PsyDialogueDataset
from lm.utils import SPECIAL_TOKENS, create_dataset_entry_qa

model_path = {
    'chatglm': 'chatglm_6b',
    'llama': 'llama_hf',
    'baichuan': 'baichuan2-7b-base',
    'bloomz': 'bloomz_mt'
}


def get_tokenizer(args):
    model_name_or_path = os.path.join(args.cache_dir, model_path[args.model_name])
    if 'glm' in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=args.cache_dir, use_fast=False)

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

    print('special_tokens_map:', tokenizer.specical_tokens_map)

    print(f'bos_token={tokenizer.bos_token}, bos_token_id={tokenizer.bos_token_id}')
    print(f'eos_token={tokenizer.eos_token}, eos_token_id={tokenizer.eos_token_id}')

    ds_entry = create_dataset_entry_qa(
        questions=['Q1', 'Q2'], answers=['A1', 'A2']
    )
    in_text = ds_entry.get_formatted(
        tokenizer.eos_token,
        use_system_tag=True,
        system_property_dropout=0,
        system_add_length=True
    )
    in_text = ''.join(in_text)

    prompter_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['Patient'])
    assistant_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['Doctor'])
    print(f'{prompter_token_id=}, {assistant_token_id=}')

    tr = tokenizer(in_text, max_length=1024, pad_to_max_length=False, truncation=True)

    message_indices = []
    i = -1
    for id in tr.inputs_ids:
        if id in (prompter_token_id, assistant_token_id):
            i += 1
        message_indices.append(i)

    print('encoding result:', tr)
    for i, xs in enumerate(tr.inputs_ids):
        decoded = tokenizer.decode(xs)
        print(f'{i}: {xs} -> {decoded}')

    print('message_indices:', message_indices)


def get_model(args):
    model_name_or_path = os.path.join(args.cache_dir, model_path[args.model_name])
    if 'glm' in model_name_or_path:
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=args.cache_dir)
    return model


def get_dataset(args, tokenizer):
    train_set = PsyDialogueDataset(args.data_dir, tokenizer)
    eval_set = PsyDialogueDataset(args.data_dir, tokenizer, data_type='eval')
    return train_set, eval_set

