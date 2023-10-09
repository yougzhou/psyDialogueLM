import argparse
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import datasets
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, set_seed, PreTrainedModel
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available
from peft import get_peft_model, LoraConfig, TaskType

from lm.utils import str2bool, print_args, compute_metrics, preprocess_logits_for_metrics
from lm.finetuning import get_loss, get_tokenizer, get_collator, get_model, get_dataset, tokenizer_sanity_check, get_metrics


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_tensorboard', type=str2bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--deepspeed', type=str2bool, default=False)
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--show_dataset_stats', type=str2bool, default=False)

    parser.add_argument('--random_seed', type=int, default=29)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=False)
    parser.add_argument('--max_length', type=int, default=2048)

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='../.cache')
    parser.add_argument('--lora', type=str2bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=400)
    parser.add_argument('--save_total_limit', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=100)
    return parser.parse_args()


class CustomTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            sampler: torch.utils.data.sampler.Sampler = None,
            loss_function: str = "CrossEntropyLoss",
            poly_eps: float = 1.0,
            train_collate_fn: Callable = None,
            **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.train_collate_fn = train_collate_fn
        # By default CrossEntropyLoss ignores padding_index -100, but just in case use our own loss_fct
        self.loss_fct = get_loss(loss_function, poly_eps)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_outputs=False):
        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return (loss, outputs) if return_outputs else loss

    def _compute_loss(self, model, inputs):
        inputs = self._prepare_inputs(inputs)

        labels_mask = inputs.pop("label_masks")
        targets = inputs.pop("targets")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            use_cache=False,
        )

        logits = outputs.get("logits")

        loss = self.loss_fct(outputs.get("logits"), targets, mask=labels_mask)

        return loss, logits, targets, labels_mask

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, logits, labels, labels_mask = self._compute_loss(model, inputs)
            labels[~labels_mask.bool()] = -100  # padding_index

        loss = loss.mean().detach()

        if self.args.prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader


def main(args):
    if not args.deepspeed or args.local_rank == 0:
        print_args(args)

    output_dir = (
        os.path.join(args.output_dir, f'{args.model_name}_finetuned') if args.output_dir else f'{args.model_name}-{args.log_dir}-finetuned'
    )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    optimizer = OptimizerNames.ADAMW_TORCH

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=float(args.learning_rate),
        deepspeed=args.deepspeed_config if args.deepspeed else None,
        optim=optimizer,
        fp16=args.dtype in ['fp16', 'float16'],
        local_rank=args.local_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.log_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to=['tensorboard'] if args.log_tensorboard else None
    )

    set_seed(args.random_seed)

    tokenizer = get_tokenizer(args)
    if not args.deepspeed or args.local_rank == 0:
        tokenizer_sanity_check(tokenizer)

    collate_fn = get_collator(
        args=args,
        tokenizer=tokenizer,
    )

    train_set, eval_set = get_dataset(args)
    metrics, preprocess_fns = get_metrics(args, tokenizer)
    model = get_model(args, tokenizer)

    if args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias='all'
        )
        model = get_peft_model(model, peft_config)
        print('training the model with lora ...')
        model.print_trainable_parameters()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        train_collate_fn=collate_fn,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=metrics, preprocess_fns=preprocess_fns),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    args = setup_args()
    if args.deepspeed:
        args.world_size = int(os.getenv('WORLD_SIZE', default=1))
    main(args)

