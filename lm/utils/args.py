import argparse


def print_args(args):
    """Print arguments"""
    print('-------------------- Arguments --------------------', flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append(f'  {arg} {dots} {getattr(args, arg)}')

    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)

    print('---------------- End of arguments -----------------', flush=True)


def str2bool(v):
    """ Support bool type for argparse. """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
