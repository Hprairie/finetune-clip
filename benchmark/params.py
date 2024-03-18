import argparse
import os


# Get args
def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--k',
            nargs='+',
            type=int,
            help='Retrieval @k'
    )
    parser.add_argument(
            '--name',
            type=str,
            help='Name of run'
    )
    parser.add_argument(
            '--model',
            type=str,
            help='Name of model architecture'
    )
    parser.add_argument(
            '--pretrained',
            type=str,
            default=None,
            help='Either a path to a local point or name from web'
    )
    parser.add_argument(
            '--altered',
            default=False,
            action='store_true',
            help='If altered architecture'
    )
    parser.add_argument(
            '--finetune-path',
            default=None,
            type=str,
            help='Path to altered architecture file'
    )
    parser.add_argument(
            '--dataset',
            type=str,
            help='Path to root'
    )
    parser.add_argument(
            '--dataset-ann',
            type=str,
            default='',
            help='Path to annotation file'
    )
    parser.add_argument(
            '--batchsize',
            type=int,
            default=32,
            help='Batch Size to use when encoding'
    )
    parser.add_argument(
            '--reg_retrieval',
            default=False,
            action='store_true',
            help='Run regular retrieval instead of finegrained'
    )
    args = parser.parse_args(args)

    # Get hyper-parameter information from log file
    pretrained_info = argparse.ArgumentParser()
    if args.pretrained is not None and args.altered:
        with open(os.path.join(args.pretrained, 'params.txt'), 'r') as file:
            for line in file:
                pass
    
    return args
