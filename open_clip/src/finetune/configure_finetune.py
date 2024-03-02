import torch
from typing import Union
import sys
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP, CustomTextCLIP
from training.params import parse_args
from .lora import register_lora_clip


def configure_model(model: Union[CLIP, CustomTextCLIP], args, logger=None) -> CLIP:
    if args.linear_probing is not None:
        """#Assumes that linear-probing is passed as a string in the following format:

        * projection:dim                    # dim is the final projected dimension.
        * mlp:hidden_dim:output_dim         # Similar to project but with non-linearity
                                            # (input_dim -> hidden_dim -> activation -> output_dim
        """
        assert any(s in args.linear_probing for s in ['projection', 'mlp'])

        linear_probing = args.linear_probing.split(':')
        probing_type, config = linear_probing[0], linear_probing[1:]
        match probing_type:
            case 'projection':
                pass
            case 'mlp':
                pass

    if args.freeze_layers is not None:
        """#Assumes that freeze_layers is passed as a string in the following format:

        * all:layers            # Note that layers is the number of blocks to keep
        * image:layers          # unlocked at the end of the model
        * text:layers
        """
        assert any(s in args.freeze for s in ['all', 'image', 'text'])

        tower, layers = args.freeze_layer.split(':')
        match tower:
            case 'all':
                model.visual.lock(unlocked_groups=layers, freeze_bn_stats=True)
                if isinstance(model, CustomTextCLIP):
                    model.text.lock(unlocked_groups=layers, freeze_bn_stats=True)
                else:
                    model.transformer.lock(unlocked_groups=layers, freeze_bn_stats=True)
            case 'image':
                model.visual.lock(unlocked_groups=layers, freeze_bn_stats=True)
            case 'text':
                if isinstance(model, CustomTextCLIP):
                    model.text.lock(unlocked_groups=layers, freeze_bn_stats=True)
                else:
                    model.transformer.lock(unlocked_groups=layers, freeze_bn_stats=True)

    if args.lora is not None:
        """#Assumes that lora is passed as a str with the following format:

        * --lora=rank:alpha

        # TODO: Add Layerwise Customization to LoRA
        """
        # Freeze Model
        for n, p in model.named_parameters():
            p.requires_grad = False

        # Add LoRa Modules
        rank, alpha = args.lora.split(':')
        register_lora_clip(model, rank=int(rank), alpha=int(alpha))

