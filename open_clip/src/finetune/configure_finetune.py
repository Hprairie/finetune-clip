import torch
from typing import Union
import sys
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import CLIP, CustomTextCLIP
from training.params import parse_args
from .lora import register_lora_clip, LoRaModule


def configure_model(model: Union[CLIP, CustomTextCLIP], args, logger=None) -> CLIP:
    """#Function which augments CLIP and CustomTextCLIP objects and adds new modules
    to finetune the model. For several features, this function will overwrite the 
    forward function of the model in order add additional functionality.

    Notes:
        -> When adding new modules to Model, will create a new feature called
        called finetune-modules, which allows for easy stripping and saving.
            -> Will also add a new feature to determine if we should save the
            base model.
        -> When trying to nest new features within model use reparamaterize
        if possible.
        -> Worst case just overwrite the forward function of the desired class
            ** Use this sparingly **

    * Linear Probing:
        -> Not IMPLEMENTED, can't come up with a nice way to to it without doing
        something super hacky. Will probably just go into existing classes and edit
        the information.

    * Freeze Layers:
        -> Working

    * LoRA:
        -> Working

    """
    if args is None:
        return

    save_full_model = False
    finetune_modules = []

    if args.linear_probing is not None:
        """#Assumes that linear-probing is passed as a string in the following format:

        * projection:dim                    # dim is the final projected dimension.
        * mlp:hidden_dim:output_dim         # Similar to project but with non-linearity
                                            # (input_dim -> hidden_dim -> activation -> output_dim
        """
        assert any(s in args.linear_probing for s in ['projection', 'mlp'])

        linear_probing = args.linear_probing.split(':')
        probing_type, config = linear_probing[0], [int(x) for x in linear_probing[1:]]
        match probing_type:
            case 'projection':
                assert NotImplementedError
            case 'mlp':
                assert NotImplementedError

    if args.freeze_layers is not None:
        """#Assumes that freeze_layers is passed as a string in the following format:

        * all:layers            # Note that layers is the number of blocks to keep
        * image:layers          # unlocked at the end of the model
        * text:layers
        """
        assert any(s in args.freeze_layers for s in ['all', 'image', 'text'])

        tower, layers = args.freeze_layers.split(':')
        layers = int(layers)
        unfreeze_norm = args.unfreeze_norm
        match tower:
            case 'all':
                model.lock_image_tower(unlocked_groups=layers, freeze_bn_stats=unfreeze_norm)
                model.lock_text_tower(unlocked_groups=layers, freeze_layer_norm=unfreeze_norm)
            case 'image':
                model.lock_image_tower(unlocked_groups=layers, freeze_bn_stats=unfreeze_norm)
            case 'text':
                model.lock_text_tower(unlocked_groups=layers, freeze_layer_norm=unfreeze_norm)

        save_full_model = True # TODO: Implement this

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
        register_lora_clip(model, rank=int(rank), alpha=float(alpha))

        # Iterate through modules and add LoRA modules to list of finetune modules
        for module in model.modules():
            if isinstance(module, LoRaModule):
                finetune_modules.append(module)

    # Register the save parameters to the model
    model.save_full_model = save_full_model
    model.fintune_modules = finetune_modules


