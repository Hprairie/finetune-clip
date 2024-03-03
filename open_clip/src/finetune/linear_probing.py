import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from open_clip.model import CLIP, CustomTextCLIP


def projection_probing(
        model: Union[CLIP, CustomTextCLIP],
        output_dim: int
        ):
    pass

def mlp_probing(
        model: Union[CLIP, CustomTextCLIP],
        output_dim: int
        ):
    pass
