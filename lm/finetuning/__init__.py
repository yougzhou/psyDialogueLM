import os

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .datasets import PsyDialogueDataset

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
    return tokenizer


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

