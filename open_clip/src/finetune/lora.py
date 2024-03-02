import torch
from torch import einsum
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from open_clip.transformer import ResidualAttentionBlock, \
        Transformer, VisionTransformer, TextTransformer, MultimodalTransformer, AttentionalPooler
from open_clip.model import CLIP, CustomTextCLIP
from open_clip.timm_model import TimmModel
from open_clip.modified_resnet import ModifiedResNet
from open_clip.hf_model import HFTextEncoder
from einops import rearrange


# =================================================================== #

class LoRaModule(nn.Module):
    def __init__(self,
                 input_dimension: int, 
                 output_dimension: int,
                 split_dimension: int = 1,
                 device: torch.device = None,
                 rank: int = 1,
                 alpha: float = 1.
    ):
        super(LoRaModule, self).__init__()
        if split_dimension != 1: # Thank you pytorch for combining kqv into one param :(
            head_dimension = input_dimension // split_dimension
            assert head_dimension * split_dimension == input_dimension, \
                "Split dimension must be a multiple of input dimension"

            # Create's a lora for each projection matrix
            self.lora_A = nn.Parameter(torch.zeros(split_dimension, rank, output_dimension))
            self.lora_B = nn.Parameter(torch.zeros(split_dimension, head_dimension, rank))
            nn.init.normal_(self.lora_A, mean=0, std=1)
            self.split_dimension = split_dimension

        else:
            # Described in Section 4.1 of the LoRA paper
            self.lora_A = nn.Parameter(torch.zeros(rank, output_dimension))
            self.lora_B = nn.Parameter(torch.zeros(input_dimension, rank))
            nn.init.normal_(self.lora_A, mean=0, std=1)
            self.split_dimension = None

        # Described in Section 4.1 of the paper
        self.scale = alpha / rank

        self.enabled = True

        if device is not None:
            self.to(device)

    def forward(self, original_weights):
        if self.enabled:
            if self.split_dimension is None:
                return original_weights + (self.lora_B @ self.lora_A).view(original_weights.shape) * self.scale
            else:
                return original_weights + self.scale * \
                        rearrange(einsum('hir,hro->hio', self.lora_B, self.lora_A), 'h i o-> (h i) o').view(original_weights.shape)
        return original_weights


# =================================================================== #

def register_lora_linear(
        l_block: nn.Linear,
        **kwargs
        ):
    register_parametrization(
            l_block, 
            "weight", 
            LoRaModule(
                *l_block.weight.shape,
                device=l_block.weight.device,
                **kwargs
            )
    )

def register_lora_multiheaded_attention(
        mh_block: nn.MultiheadAttention, 
        **kwargs
        ):
    """#Note:

    * Bias is ignored here as bias projection in nn.MultiheadAttention
      is one-dimensional.
    """
    # Reparamaterize the weights for q,k,v matrices
    if mh_block.in_proj_weight is not None:
        register_parametrization(
                mh_block, 
                "in_proj_weight", 
                LoRaModule(
                    *mh_block.in_proj_weight.shape,
                    split_dimension=3,
                    device = mh_block.in_proj_weight.device,
                    **kwargs
                )
        )
    else:
        register_parametrization(
                mh_block, 
                "q_proj_weight", 
                LoRaModule(
                    *mh_block.q_proj_weight.shape,
                    device = mh_block.q_proj_weight.device,
                    **kwargs
                )
        )
        register_parametrization(
                mh_block, 
                "k_proj_weight", 
                LoRaModule(
                    *mh_block.k_proj_weight.shape,
                    device = mh_block.k_proj_weight.device,
                    **kwargs
                )
        )
        register_parametrization(
                mh_block, 
                "v_proj_weight", 
                LoRaModule(
                    *mh_block.v_proj_weight.shape,
                    device = mh_block.v_proj_weight.device,
                    **kwargs
                )
        )

def register_lora_mlp(
        mlp_block: nn.Sequential,
        **kwargs
        ):
    # Register LoRA to all linear layers
    for module in mlp_block.modules():
        if isinstance(module, nn.Linear):
            register_lora_linear(module, **kwargs)


def register_lora_residual_attention_block(
        ra_block: ResidualAttentionBlock,
        **kwargs
        ):
    # Registers Attention and MLP block
    register_lora_multiheaded_attention(ra_block.attn, **kwargs)
    register_lora_mlp(ra_block.mlp, **kwargs)


def register_lora_attetional_layer(
        al_block: AttentionalPooler,
        **kwargs
        ):
    register_parametrization(
            al_block, 
            'query',
            LoRaModule(
                *al_block.query.shape,
                device = al_block.query.device
                **kwargs
            )
    )
    register_lora_multiheaded_attention(al_block.attn, **kwargs)

def register_lora_transformer(
        transformer: Transformer,
        **kwargs
        ):
    for module in transformer.resblocks.children():
        register_lora_residual_attention_block(module, **kwargs)

def register_lora_vision_transformer(
        v_transformer: VisionTransformer,
        **kwargs
        ):
    # Note that there are parameters which have no been lorified
    #   - Learned Positional Embeddings
    #   - Projection Embeddings for Patches

    register_lora_transformer(v_transformer.transformer, **kwargs)

    if v_transformer.attn_pool is not None:
        register_lora_attetional_layer(v_transformer.attn_pool, **kwargs)
    if v_transformer.proj is not None:
        register_parametrization(
                v_transformer, 
                'proj', 
                LoRaModule(
                    *v_transformer.proj.shape,
                    device = v_transformer.proj.device,
                    **kwargs
                )
        )

def register_lora_text_transformer(
        t_transformer: TextTransformer,
        **kwargs
        ):
    register_lora_transformer(t_transformer.transformer, **kwargs)
    if isinstance(t_transformer.text_projection, nn.Linear):
        register_lora_linear(t_transformer.text_projection, **kwargs)
    else:
        register_parametrization(
                t_transformer, 
                "text_projection", 
                LoRaModule(
                    *t_transformer.text_projection.shape,
                    device = t_transformer.text_projection.device,
                    **kwargs
                )
        )

def register_lora_multimodal_transformer(
        mm_transformer: MultimodalTransformer,
        **kwargs,
        ):
    for module in mm_transformer.cross_attn.children():
        register_lora_residual_attention_block(module, **kwargs)
    register_parametrization(
            mm_transformer, 
            "text_projection", 
            LoRaModule(
                *mm_transformer.text_projection.shape,
                device = mm_transformer.text_projection.device,
                **kwargs
            )
    )

# =================================================================== #

def register_lora_vision_tower(
        v_tower: Union[TimmModel,VisionTransformer,ModifiedResNet],
        **kwargs
        ):
    if isinstance(v_tower, VisionTransformer):
        register_lora_vision_transformer(v_tower, **kwargs)
    if isinstance(v_tower, TimmModel):
        raise NotImplementedError
    if isinstance(v_tower, ModifiedResNet):
        raise NotImplementedError

def register_lora_text_tower(
        t_tower: Union[HFTextEncoder, TextTransformer],
        **kwargs
        ):
    if isinstance(t_tower, TextTransformer):
        register_lora_text_transformer(t_tower, **kwargs)
    if isinstance(t_tower, HFTextEncoder):
        raise NotImplementedError

def register_lora_clip(
        clip_model: Union[CLIP, CustomTextCLIP],
        **kwargs
        ):
    if isinstance(clip_model, CLIP):
        register_lora_vision_tower(clip_model.visual, **kwargs)
        register_lora_transformer(clip_model.transformer, **kwargs)
        if isinstance(clip_model.text_projection, nn.Linear):
            register_lora_linear(clip_model.text_projection, **kwargs)
        else:
            register_parametrization(
                    clip_model, 
                    "text_projection", 
                    LoRaModule(
                        *clip_model.text_projection.shape,
                        device = clip_model.text_projection.device,
                        **kwargs
                    )
            )
    if isinstance(clip_model, CustomTextCLIP):
        register_lora_vision_tower(clip_model.visual, **kwargs)
        register_lora_text_tower(clip_model.text, **kwargs)

