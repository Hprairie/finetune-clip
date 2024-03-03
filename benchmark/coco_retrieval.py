import hnswlib
import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import clip
import collections
import json
import argparse

# Change these to path of local COCO dataset:
coco_root = "$DATASETS/coco/images/val2017"
coco_ann_file = "$DATASETS/coco/annotations/captions_val2017.json"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model, transform = clip.load("ViT-B/32")
model.to(device).eval()

with open(coco_ann_file, 'r') as f:
    data = json.load(f)
    
    list_image_path = collections.defaultdict(int)
    for i, image in enumerate(data["images"]):
        list_image_path[image["file_name"]] = i
        
    print(json.dumps(list_image_path, indent=4))
        

dataset = CocoCaptions(
    root=coco_root,
    annFile=coco_ann_file,
    transform=transform,
    # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    target_transform=lambda texts: clip.tokenize(texts[:5])
)

# k_vals=[1, 5, 10, 50]
k_vals=[5]

# Encodes all text and images in a dataset
@torch.no_grad()
def encode_dataset(clip, dataset: dutils.Dataset, batch_size = 16):
    # image_to_text_map[i] gives the corresponding text indices for the ith image
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    image_encodings = []
    text_encodings = []

    text_index = 0
    image_index = 0
    
    patch_to_image_map = []
    
    text_to_encoding_map = {}
    
    for images, text in dataloader:
        images = images.to(device)
        text = text.to(device)
        
        # text has shape B x 5 x 77
        batch_size, captions_per_image, _ = text.shape
        # B x 5 x 77 -> (B*5) x 77
        text = torch.flatten(text, start_dim=0, end_dim=1)

        patch_representations = clip.encode_image(images)
        text_features = clip.encode_text(text)
        
        for i, encoding in enumerate(text_features):
            text_to_encoding_map[text[i]] = encoding

        image_encodings.append(patch_representations)
        text_encodings.append(text_features)

        # Update text_to_image_map and image_to_text_map for this batch
        for i in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image

            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image

            patches_per_image = patch_representations.shape[1]
            patch_to_image_map += [image_index] * patches_per_image
            
            image_index += 1

    image_encodings = torch.cat(image_encodings)
    text_encodings = torch.cat(text_encodings)
    
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)
    patch_to_image_map = torch.LongTensor(patch_to_image_map).to(device)

    # Normalize encodings
    image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
    text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

    return image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map


def recall_at_k(clip, dataset: dutils.Dataset, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map = encode_dataset(clip, dataset, batch_size=batch_size)
    
    image_encodings = torch.cat([patches for patches in image_encodings], dim=0)
    text_encodings = torch.cat([text for text in text_encodings], dim=0)
 
    num_text = len(text_to_image_map)
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    
    print(image_encodings.shape)

    # Create a hnswlib index for the image embeddings
    dim = image_encodings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_im, ef_construction=250, M=16)
    p.set_ef(50)
    p.add_items(image_encodings.cpu().numpy())

    # text-to-image recall
    print("Text-to-image recall...")
    text_to_image_recall = []
    n_patches = 50

    for k in k_vals:
        correct_recall_count = 0
        
        for caption_idx in range(num_text):
            token_to_image_scores = {}
            
            # Get the text encodings for this caption
            caption_text_encodings = text_encodings[caption_idx * 77 : (caption_idx + 1) * 77]
            
            # Get the position of the EOT token which is the highest number in the sequence
            eot_idx = torch.argmax(caption_text_encodings[:, -1])
    
            assert eot_idx <= 77 
            
            # print(eot_idx)
            
            # For each token in the caption until eot_idx
            for token_idx in range(caption_idx * 77, caption_idx * 77 + eot_idx):
            # for token_idx in range(caption_idx * 77, (caption_idx + 1) * 77):
                token_to_image_scores[token_idx] = collections.defaultdict(float)
                
                # Query the kNN for n_patches closest patches
                labels, distances = p.knn_query(text_encodings[token_idx].cpu().unsqueeze(0), k=n_patches)
                labels, distances = labels[0], distances[0]
                
                # Map each patch to its parent image and calculate cosine similarity
                for label, distance in zip(labels, distances):
                    parent_image = patch_to_image_map[label].item()
                    cos_sim = 1 - distance
                    
                    cur_max = token_to_image_scores[token_idx][parent_image]
                    
                    # Store max similarity per image
                    token_to_image_scores[token_idx][parent_image] = max(cur_max, cos_sim)
                    
            # Sum the cosine similarities across every token in the caption
            image_scores_summed = collections.defaultdict(float)
            for image_scores in token_to_image_scores.values():
                for image, score in image_scores.items():
                    image_scores_summed[image] += score

            # print(image_scores_summed)

            # Get the top k images by score
            top_k_images = sorted(image_scores_summed.keys(), key=lambda image: image_scores_summed[image], reverse=True)[:k]

            # Check if the correct image is in the top k images
            correct_image = text_to_image_map[caption_idx].item()
            # print(top_k_images, correct_image)
            if correct_image in top_k_images:
                correct_recall_count += 1

        # Compute recall for this k
        # print(correct_recall_count)
        recall = correct_recall_count / num_text
        text_to_image_recall.append(recall)

    return text_to_image_recall

if __name__ == '__main__':
    t2i = recall_at_k(model, dataset, k_vals=k_vals, batch_size=16)

    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")
