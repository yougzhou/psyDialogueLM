import argparse
import os

from lm.evaluation import get_evaluator


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=['single_choice', 'multi_choice', 'case_qa', 'psychological_dialogue'])
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()


def main(args):
    data_path = os.path.join(args.data_dir, args.subject, 'test.json')
    evaluator = get_evaluator(args)


if __name__ == '__main__':
    print('-------------------- evaluate script --------------------')
    args = setup_args()
    main(args)
