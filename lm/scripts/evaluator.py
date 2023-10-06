import argparse
import os

from lm.evaluation import get_evaluator
from lm.utils import print_args, AutoEvaluator, read_json


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, choices=['single_choice', 'multi_choice', 'case_qa', 'dialogue'])
    parser.add_argument('--evaluator', type=str, choices=['marking', 'referee'])
    parser.add_argument('--evaluator_model', type=str, default='gpt3.5-turbo')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--ori_key', type=str, default='')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--cache_dir', type=str, default='../.cache')
    return parser.parse_args()


def main(args):
    print_args(args)
    evaluator = get_evaluator(args)
    auto_evaluator = AutoEvaluator(args)
    if 'single_choice' in args.subject:
        data_path = os.path.join(args.data_dir, args.subject, 'test.csv')
        save_path = os.path.join(args.save_dir, args.subject, f'{args.model_name}_result.csv')
        result = evaluator.eval_subject(data_path, save_path)
    elif 'dialogue' in args.subject:
        data_path = os.path.join(args.data_dir, args.subject, 'test.json')
        save_path = os.path.join(args.save_dir, args.subject, f'{args.model_name}_result.json')
        result = evaluator.eval_dialogue(data_path, save_path)
    elif 'case_qa' in args.subject:
        data_path = os.path.join(args.data_dir, args.subject, 'test.csv')
        save_path = os.path.join(args.save_dir, args.subject, f'{args.model_name}_result.json')
        if os.path.exists(save_path):
            result = read_json(save_path)
        else:
            result = evaluator.eval_case(data_path, save_path)
        result = auto_evaluator(result)
    else:
        raise NotImplementedError
    print(result)


if __name__ == '__main__':
    args = setup_args()
    main(args)
