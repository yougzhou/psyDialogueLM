import argparse
import math

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, set_seed
from peft import get_peft_model, LoraConfig, TaskType

from lm.utils import str2bool, print_args, SPECIAL_TOKENS, compute_metrics
from lm.finetuning import get_tokenizer, get_model, get_dataset


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='../.cache')
    parser.add_argument('--lora', type=str2bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=400)
    parser.add_argument('--log_steps', type=int, default=100)
    return parser.parse_args()


def main(args):
    print_args(args)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        deepspeed=args.deepspeed_config if args.deepspeed else None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='steps',
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_steps=args.save_steps,
        save_total_limit=4,
        eval_steps=args.eval_steps,
        logging_steps=args.log_steps,
        report_to=['tensorboard']
    )

    set_seed(args.random_seed)

    tokenizer = get_tokenizer(args)
    tokenizer.add_special_tokens({
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token
    })
    additional_special_tokens = (
        [] if 'additional_special_tokens' not in tokenizer.special_tokens_map else tokenizer.special_tokens_map['additional_special_tokens']
    )
    additional_special_tokens = list(set(additional_special_tokens + list(SPECIAL_TOKENS.values())))
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    model = get_model(args)
    num_embeddings = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != num_embeddings:
        p = 16
        target_size = math.ceil(len(tokenizer) / p) * p
        model.resize_token_embeddings(target_size)

    train_set, eval_set = get_dataset(args, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        max_length=2048,
        padding='longest'
    )

    if args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias='all'
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    args = setup_args()
    main(args)

