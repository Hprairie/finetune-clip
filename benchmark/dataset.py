from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
import torch
from typing import List, Union
import collections
import json
import argparse
from open_clip.model import CLIP, CustomTextCLIP
from einops import rearrange
from tqdm import tqdm


# Encodes all text and images in a dataset
@torch.no_grad()
def encode_dataset(
        clip: Union[CLIP, CustomTextCLIP],
        dataset: dutils.Dataset,
        batch_size: int = 16,
        reg_retrieval: bool = False,
        mask_padding: bool = False,
        second_to_last: bool = False
    ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    clip.to(device)

    # image_to_text_map[i] gives the corresponding text indices for the ith image
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    image_encodings = []
    text_encodings = []
    all_masks = []

    text_index = 0
    image_index = 0
    
    patch_to_image_map = []
    
    text_to_encoding_map = {}

    dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for images, texts_and_masks in tqdm(dataloader):
        images = images.to(device)
        text = texts_and_masks['tokens'].to(device)
        if mask_padding:
            masks = texts_and_masks['mask'].to(device)
            masks = rearrange(masks, 'b s -> (b s)')
        
        # B x 5 x 77 -> (B*5) x 77
        batch_size, captions_per_image, _ = text.shape
        text = rearrange(text, 'b s e -> (b s) e')

        if reg_retrieval:
            image_embeddings = clip.encode_image(images)
            text_embeddings = clip.encode_text(text)
        else:
            _, image_embeddings = clip.encode_image(images, return_tokens=True, second_to_last=second_to_last)
            _, text_embeddings = clip.encode_text(text, return_tokens=True, second_to_last=second_to_last)
        
        # TODO: get rid of text_to_encoding_map
        # if not reg_retrieval:
        #     for i, encoding in enumerate(text_embeddings):
        #         text_to_encoding_map[text[i]] = encoding

        image_encodings.append(image_embeddings)
        text_encodings.append(text_embeddings)
        
        if mask_padding:
            all_masks.append(masks)

        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image

            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image

            if not reg_retrieval:
                patches_per_image = image_embeddings.shape[1]
                patch_to_image_map += [image_index] * patches_per_image
            
            image_index += 1

    image_encodings = torch.cat(image_encodings)
    text_encodings = torch.cat(text_encodings)
    
    if mask_padding:
        all_masks = torch.cat(all_masks)
    
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)
    
    if not reg_retrieval:
        patch_to_image_map = torch.LongTensor(patch_to_image_map).to(device)

    # Normalize encodings (Why are we normalizing?)
    #image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    #text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

    return image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map, all_masks

def get_dataset(args, transform, tokenizer, mask_padding, repeat_tokens):
    dataset = CocoCaptions(
        root=args.dataset,
        annFile=args.dataset_ann,
        transform=transform,
        # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
        target_transform=lambda texts: tokenizer(texts[:5], return_mask=mask_padding, repeat_tokens=repeat_tokens),
    )
    return dataset
