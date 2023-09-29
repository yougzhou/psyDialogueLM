import argparse
import os

from lm.evaluation import get_evaluator
from lm.utils import print_args


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, choices=['single_choice', 'multi_choice', 'case_qa', 'psy_dialog'])
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./outputs')
    parser.add_argument('--cache_dir', type=str, default='../.cache')
    return parser.parse_args()


def main(args):
    print_args(args)
    if 'choice' in args.subject:
        data_path = os.path.join(args.data_dir, args.subject, 'test.csv')
        save_path = os.path.join(args.save_dir, args.subject, f'{args.model_name}_result.csv')
    else:
        data_path = os.path.join(args.data_dir, args.subject, 'test.json')
        save_path = os.path.join(args.save_dir, args.subject, 'result.json')
    evaluator = get_evaluator(args)
    result = evaluator.eval_subject(data_path, save_path)
    print(result)


if __name__ == '__main__':
    args = setup_args()
    main(args)
