import os.path
import re
from collections import defaultdict
from time import sleep

import evaluate
from tqdm import tqdm

import openai
import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

from .format import read_txt


def compute_metrics(eval_pred, preprocess_fns, metrics):
    outputs = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        outputs = dict(**outputs, **metric.compute(predictions=preds, references=labels))
    return outputs


def default_preprocess(eval_pred, ignore_negative_labels=True):
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    if not ignore_negative_labels:
        return preds, labels

    mask = labels > 0
    return preds[mask], labels[mask]


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class MetricComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.right = torch.Tensor([0])
        self.total = torch.Tensor([0])

    def format_pred(self, inputs):
        response, target = [], []
        for sample in inputs:
            response.append(sample['response'])
            target.append(sample['target'])
        assert len(target) == len(response)
        return response, target

    def strip_special_tokens(self, lines):
        outputs = [re.sub(r'<[^>]+>', '', s) for s in lines]
        outputs = [re.sub('doctor：', '', s).strip() for s in outputs]
        return outputs

    def post_process_text(self, response, target):
        return self.strip_special_tokens(response), self.strip_special_tokens(target)

    def compute_rouge(self, response, target, avg=False, ignore_empty=False):
        rouge_score = Rouge().get_scores(response, target, avg=avg, ignore_empty=ignore_empty)
        return rouge_score['rouge-l']['f']

    def compute_bleu(self, response, target, n=1):
        return np.average([sentence_bleu([target], response, weights=tuple(1 / n for _ in range(n)))])

    def compute_meteor(self, response, target):
        return np.average([meteor_score([t.split()], r.split()) for t, r in zip(target, response)])

    def compute_entropy(self, generated):
        etp_score = [0.0, 0.0, 0.0, 0.0]
        div_score = [0.0, 0.0, 0.0, 0.0]
        counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
        for gg in generated:
            g = gg.rstrip().split()
            for n in range(4):
                for idx in range(len(g) - n):
                    ngram = " ".join(g[idx: idx + n + 1])
                    counter[n][ngram] += 1
        for n in range(4):
            total = sum(counter[n].values()) + 1e-10
            for v in counter[n].values():
                etp_score[n] += -(v + 0.0) / total * (np.log(v + 0.0) - np.log(total))
            div_score[n] = (len(counter[n].values()) + 0.0) / total
        return div_score[2]

    def compute(self, predictions, references):
        self.right += (predictions == references).masked_fill(references.eq(-100), 0).sum().item()
        self.total += (references != -100).sum().item()
        return {'accuracy': (self.right / self.total).item()}

    def __call__(self, inputs):
        response, target = self.format_pred(inputs)
        result = dict()
        response, target = self.post_process_text(response, target)
        response, target = [' '.join(self.tokenizer.tokenize(s)) for s in response], [' '.join(self.tokenizer.tokenize(s)) for s in target]
        result['rouge-l'] = self.compute_rouge(response, target, avg=True, ignore_empty=True)
        result['bleu-2'] = self.compute_bleu(response, target, n=2)
        result['bleu-4'] = self.compute_bleu(response, target, n=4)
        result['meteor'] = self.compute_meteor(response, target)
        result['dist-2'] = self.compute_entropy(response)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


class AutoEvaluator:
    def __init__(self, args):
        self.evaluator = args.evaluator
        self.model_name = args.evaluator_model
        openai.api_key = args.api_key
        openai.ori_key = args.ori_key
        self.prompt_path = os.path.join('./packages/prompts', args.evaluator)

    def preprocess(self, sample, prompt):
        response, answer = sample['response'], sample['target']
        prompt = prompt.replace('[Answer]', answer)
        prompt = prompt.replace('[Response]', response)
        prompt = [{'role': 'user', 'content': prompt}]
        return prompt

    def evaluate_score(self, inputs):
        result = []
        prompt = read_txt(self.prompt_path)
        for sample in tqdm(inputs):
            prompt = self.preprocess(sample, prompt)
            response = None
            timeout_count = 0
            while response is None and timeout_count <= 30:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=prompt,
                        temperature=0.
                    )
                except Exception as msg:
                    if 'timeout' in str(msg):
                        timeout_count += 1
                    sleep(5)
                    continue
            if response is None:
                response_str = ''
            else:
                response_str = response['choices'][0]['message']['content']
            response_str = response_str.strip()
            if len(response_str) > 0:
                score = self.extract_score(response_str)
            else:
                score = 0
            result.append(score)
        avg_score = sum(result) / len(result)
        return avg_score

    def extract_score(self, response_str):
        pass

    def __call__(self, inputs):
        if self.evaluator == 'marking':
            score = self.evaluate_score(inputs)
            print(f'Average score: {score}')
        else:
            pass
