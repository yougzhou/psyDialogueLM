from .core import GLMEvaluator

def get_evaluator(args):
    if 'glm' in args.model_name:
        evaluator = GLMEvaluator(args)
    elif 'turbo' in args.model_name:
        evaluator = ChatGPTEvaluator(args)
    elif 'llama' in args.model_name:
        evaluator = LLaMAEvaluator(args)
    elif 'bloomz' in args.model_name:
        evaluator = BloomzEvaluator(args)
    elif 'baichuan' in args.model_name:
        evaluator = BaiChuanEvaluator(args)
    elif 'educhat' in args.model_name:
        evaluator = EduChatEvalutor(args)
    elif 'qianwen' in args.model_name:
        evaluator = QianWenEvaluator(args)
    elif 'yiyan' in args.model_name:
        evaluator = YiYanEvaluator(args)
    elif 'xinghuo' in args.model_name:
        evaluator = XingHuoEvaluator(args)
    else:
        raise ValueError(f'Invalid model name {args.model_name}')
    return evaluator
