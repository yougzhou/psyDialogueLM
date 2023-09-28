class Evaluator:
    def __init__(self, args):
        self.model_name = args.model_name
        self.cache_dir = args.cache_dir


class GLMEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
