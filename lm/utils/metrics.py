import re
from collections import defaultdict

import numpy as np
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
# from nltk.translate.nist_score import sentence_nist
from rouge import Rouge


def compute_metrics(eval_pred, preprocess_fns, metrics):
    outputs = {}
    for metric, preprocess_fn in zip(metrics, preprocess_fns):
        preds, labels = preprocess_fn(eval_pred)
        outputs = dict(**outputs, **metric.compute(predictions=preds, references=labels))
    return outputs


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class MetricComputer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
