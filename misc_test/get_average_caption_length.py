import argparse
import sys
import torch
import einops
from tqdm import tqdm
from open_clip import get_tokenizer, create_model_and_transforms
from training.data import get_data
from finetune.params import parse_args

args = sys.argv[1:]
args = parse_args(args)
args.distributed = False

_, preprocess_train, preprocess_val = create_model_and_transforms(
    args.model,
    args.pretrained,
    precision=args.precision,
    device=torch.device("cpu"),
    jit=args.torchscript,
    force_quick_gelu=args.force_quick_gelu,
    force_custom_text=args.force_custom_text,
    force_patch_dropout=args.force_patch_dropout,
    force_image_size=args.force_image_size,
    image_mean=args.image_mean,
    image_std=args.image_std,
    image_interpolation=args.image_interpolation,
    image_resize_mode=args.image_resize_mode,  # only effective for inference
    aug_cfg=args.aug_cfg,
    pretrained_image=args.pretrained_image,
    output_dict=True,
    finetune_args=args, # This is lazy but idc
)
tokenizer = get_tokenizer(args.model)
data = get_data(
    args,
    (preprocess_train, preprocess_val),
    epoch=0,
    tokenizer=tokenizer,
    mask_padding=True,
)

count = 0
with tqdm(data['train'].dataloader, unit="batch") as tdata:
    for _, _, mask in tdata:
        mask = einops.rearrange(mask, 'b s -> (b s)').sum(dim=-1)
        count += mask.item()
        tdata.set_postfix(batch_average=(mask.item() / args.batch_size))

print(count / data['train'].dataloader.num_samples)

