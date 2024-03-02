import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from open_clip.model import CLIP, CustomTextCLIP


def projection_probing(
        model: Union[CLIP, CustomTextCLIP],
        output_dim: int
        ):
    # Kinda Hacky, but just swapping the final project parameter
    pass
