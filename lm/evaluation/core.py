import os
import re
from tqdm import tqdm
import requests

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from lm.utils import read_json, read_csv


class Evaluator:
    def __init__(self, args):
        self.choices = ['A', 'B', 'C', 'D']
        self.subject = args.subject
        self.model_name = args.model_name
        self.cache_dir = args.cache_dir

    def eval_subject(self, data_path, save_path):
        correct_num = 0
        if save_path:
            result = []
            score = []
        test_data = read_csv(data_path)
        answers = list(test_data['answer'])
        for row_index, row in tqdm(test_data.iterrows(), total=len(test_data)):
            question = self.format_example(row)
            inputs = self.tokenizer(question, return_tensors='pt').to('cuda')
            max_length=2048
            response = self.model.generate(**inputs, max_length=max_length)
            response_str = self.tokenizer.decode(response[0]).strip()
            if len(response_str) > 0:
                answer_list = self.extract_answer(response_str)
                if len(answer_list) > 0 and (answer_list[-1] == row['answer']):
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
            else:
                correct = 0
            if save_path:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)
        if save_path:
            test_data['model_output'] = result
            test_data['correctness'] = score
            test_data.to_csv(save_path, encoding='utf-8', index=False)
        return correct_ratio

    def format_example(self, line):
        add_prompt = '请你扮演中文心理咨询师，以下是中国关于心理咨询师等级考试的单项选择题，请选出其中的正确答案。\n'
        example = add_prompt + line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        return example

    def extract_answer(self, response_str):
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list = []
        if response_str[0] in ["A", 'B', 'C', 'D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list) == 0:
                ans_list = re.findall(p, response_str)
            else:
                break
        return ans_list


class GLMEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.model_name_or_path = os.path.join(self.cache_dir, 'chatglm-6b')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True, cache_dir=self.cache_dir).half().cuda()


class LLaMAEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.model_name_or_path = os.path.join(self.cache_dir, 'llama_hf')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).half().cuda()

class BloomzEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.mdoel_name_or_path = os.path.join(self.cache_dir, 'bloomz_mt')
        self.tokenizer = AutoTokenizer.from_pretrained(self.mdoel_name_or_path, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.mdoel_name_or_path, cache_dir=self.cache_dir).half().cuda()


class BaiChuanEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.model_name_or_path = os.path.join(self.cache_dir, 'baichuan2-7b-base')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True, cache_dir=self.cache_dir).half().cuda()


class EduChatEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.api = 'http://101.132.187.81:5000/chat'

    def eval_subject(self, data_path, save_path):
        correct_num = 0
        if save_path:
            result = []
            score = []
        test_data = read_csv(data_path)
        answers = list(test_data['answer'])
        for row_index, row in tqdm(test_data.iterrows(), total=len(test_data)):
            question = self.format_example(row)
            inputs = {
                'messages': [{"role":"system","content":"请问有什么可以帮助您的吗？"},{"role":"user","content":question}],
                'functionUse': 'chat'
            }
            response_str = requests.post(self.api, json=inputs).json()['response']
            if len(response_str) > 0:
                answer_list = self.extract_answer(response_str)
                if len(answer_list) > 0 and (answer_list[-1] == row['answer']):
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
            else:
                correct = 0
            if save_path:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100 * correct_num / len(answers)
        if save_path:
            test_data['model_output'] = result
            test_data['correctness'] = score
            test_data.to_csv(save_path, encoding='utf-8', index=False)
        return correct_ratio
