import argparse
import os
from finetune.params import parse_args as training_parse_args


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
            '--finetune-path',
            default=None,
            type=str,
            help='Path to altered architecture log'
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
    local_args = parser.parse_args(args)
    print(args)

    # Get hyper-parameter information from log file
    finetune_params = []

    unknown_vars = ['checkpoint_path', 'device', 'distill', 'distributed', 'local_rank', 'log_level', 'log_path', 'rank', 'tensorboard', 'tensorboard_path', 'wandb', 'world_size']

    if local_args.pretrained is not None and local_args.finetune_path is not None:
        with open(os.path.join(local_args.finetune_path, 'params.txt'), 'r') as file:
            for line in file:
                name, val = line.strip().split(':', 1)
                if name in unknown_vars:
                    continue
                name = '--' + name.replace('_', '-')
                val = val.strip()
                if val in ['{}', 'None', '', 'False']:
                    continue
                if val == 'True':
                    finetune_params.append(name)
                else:
                    finetune_params += [name, val]
    print(finetune_params)

    finetune_params = training_parse_args(finetune_params)
    
    local_args.finetune_args = finetune_params
    return local_args
