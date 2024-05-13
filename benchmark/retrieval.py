import hnswlib
import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import logging
import collections
import matplotlib.pyplot as plt

def recall_at_k(
        image_encodings,
        text_encodings,
        text_to_image_map,
        image_to_text_map,
        patch_to_image_map,
        text_to_encoding_map,
        masks,
        k_vals,
        batch_size,
        reg_retrieval,
        context_length = None,
    ):

    if reg_retrieval:
        return reg_recall_at_k(
            image_encodings,
            text_encodings,
            text_to_image_map,
            image_to_text_map,
            k_vals,
            batch_size,
        )
        
    return finegrained_recall_at_k(
        image_encodings,
        text_encodings,
        text_to_image_map,
        image_to_text_map,
        patch_to_image_map,
        text_to_encoding_map,
        masks,
        k_vals,
        batch_size,
        context_length,
    )
        

def reg_recall_at_k(
        image_encodings,
        text_encodings,
        text_to_image_map,
        image_to_text_map,
        k_vals,
        batch_size
    ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    
    # Create a hnswlib index for the image embeddings
    dim = image_encodings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_im, ef_construction=200, M=16)
    p.set_ef(50)
    p.add_items(image_encodings.cpu().numpy())

    text_to_image_recall = []

    for k in k_vals:
        labels, distances = p.knn_query(text_encodings.cpu().numpy(), k=k)
        labels = torch.tensor(labels.astype('int64')).to(device)

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(labels, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)
        
    return text_to_image_recall

def finegrained_recall_at_k(
        image_encodings,
        text_encodings,
        text_to_image_map,
        image_to_text_map,
        patch_to_image_map,
        text_to_encoding_map,
        masks,
        k_vals,
        batch_size,
        context_length = None,
    ):
    context_length = context_length if context_length is not None else 77
    
    image_encodings = torch.cat([patches for patches in image_encodings], dim=0)
    text_encodings = torch.cat([text for text in text_encodings], dim=0)

    num_text = len(text_to_image_map)
    num_im = image_encodings.shape[0]
    
    print(image_encodings.shape)
    print(text_encodings.shape)

    # Create a hnswlib index for the image embeddings
    dim = image_encodings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_im, ef_construction=250, M=16)
    p.set_ef(50)
    p.add_items(image_encodings.cpu().numpy())

    # text-to-image recall
    print("Text-to-image recall...")
    text_to_image_recall = []

    for k in k_vals:
        n_patches = k
        correct_recall_count = 0
        
        for caption_idx in range(num_text):
            token_to_image_scores = {}
            
            # Get the text encodings for this caption
            caption_text_encodings = text_encodings[caption_idx * context_length : (caption_idx + 1) * context_length]
            

            # Images that we want to run full maxsim on
            image_matches = set()
            
            # For each token in the caption excluding padding tokens
            for token_idx in range(caption_idx * context_length, (caption_idx + 1) * context_length):
                if masks is not None and masks[caption_idx][token_idx - (caption_idx * context_length)] == 0:
                    break

                # Query the kNN for n_patches closest patches
                labels, distances = p.knn_query(text_encodings[token_idx].cpu().unsqueeze(0), k=n_patches)
                labels, distances = labels[0], distances[0]

                # Map each path to its parent and add it to set
                for label in labels:
                    parent_image = patch_to_image_map[label].item()

                    # Add the parent_image to the set of working images
                    image_matches.add(parent_image)
                    
            # For all found images calculate the similarity scores
            top_k_images = collections.defaultdict(float)
            for match in image_matches: # <- Parallelize this and increase batch_size
                # Get patch embeddings
                patches = image_encodings[match * 49 : (match + 1) * 49]
                
                # Calculate Cosine similarity
                scaled_text_embeddings = caption_text_encodings / torch.norm(caption_text_encodings, p=2, dim=-1, keepdim=True)
                scaled_image_embeddings = patches / torch.norm(patches, p=2, dim=-1, keepdim=True)
    
                cos_sim = scaled_text_embeddings @ scaled_image_embeddings.T
                
                # Max pool per embedding
                if masks is not None:
                    score = torch.max(cos_sim, dim=1).values.mean()
                else:
                    score = torch.max(cos_sim, dim=1).values.sum()

                top_k_images[match] = score

            # Get the top k images by score
            top_images = sorted(top_k_images.keys(), key=lambda image: top_k_images[image], reverse=True)[:k]

            # Check if the correct image is in the top k images
            correct_image = text_to_image_map[caption_idx].item()
            if correct_image in top_images:
                correct_recall_count += 1

        # Compute recall for this k
        print(correct_recall_count)
        recall = correct_recall_count / num_text
        text_to_image_recall.append(recall)

    return text_to_image_recall

def reranker_recall_at_k(
        coarse_image_encodings,
        coarse_text_encodings,
        fg_image_encodings,
        fg_text_encodings,
        text_to_image_map,
        image_to_text_map,
        masks,
        k_vals,
        batch_size,
        k_multiple,
        context_length = None,
        images = None
    ):
    context_length = context_length if context_length is not None else 77

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fg_image_encodings = torch.cat([patches for patches in fg_image_encodings], dim=0)
    fg_text_encodings = torch.cat([text for text in fg_text_encodings], dim=0)
    
    num_text = coarse_text_encodings.shape[0]
    num_im = coarse_image_encodings.shape[0]
    
    # Create a hnswlib index for the image embeddings
    dim = coarse_image_encodings.shape[1]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_im, ef_construction=200, M=16)
    p.set_ef(50)
    p.add_items(coarse_image_encodings.cpu().numpy())

    # text-to-image recall
    print("Text-to-image recall...")
    print("Using Masks") if masks is not None else print("Not Using Masks")
    text_to_image_recall = []
    
    print("image encodings: ", fg_image_encodings.shape)
    print("text encodings: ", fg_text_encodings.shape)
    import pdb;pdb.set_trace()

    for k in k_vals:
        correct_recall_count = 0
        for caption_idx in range(num_text):
            
            labels, distances = p.knn_query(coarse_text_encodings[caption_idx].cpu().numpy(), k=int(k*k_multiple))
            image_matches = set(labels[0])
            
            # Plot the initial matches
            plt.figure(figsize=(10, 2))
            for i, img_idx in enumerate(image_matches):
                plt.subplot(1, len(image_matches), i + 1)
                plt.imshow(images[img_idx].permute(1, 2, 0))  # Assuming images are in CxHxW format
                plt.title(f"Initial: {img_idx}")
                plt.axis('off')
            plt.savefig(f"initial_matches_caption_{caption_idx}.png")
            plt.close()

            # For all found images calculate the similarity scores
            top_k_images = collections.defaultdict(float)
            caption_text_encodings = fg_text_encodings[caption_idx * context_length : (caption_idx + 1) * context_length]
            if masks is not None:
                caption_mask = masks[caption_idx * context_length : (caption_idx + 1) * context_length]
                caption_text_encodings = caption_text_encodings * caption_mask.unsqueeze(-1)

            for match in image_matches:
                match = int(match)

                # Get patch embeddings
                patches = fg_image_encodings[match * 49 : (match + 1) * 49]
     
                # Calculate Cosine similarity
                scaled_text_embeddings = caption_text_encodings / torch.norm(caption_text_encodings, p=2, dim=-1, keepdim=True)
                scaled_image_embeddings = patches / torch.norm(patches, p=2, dim=-1, keepdim=True)

                cos_sim = scaled_text_embeddings @ scaled_image_embeddings.T
                
                # Max pool per embedding
                score = torch.max(cos_sim, dim=1).values.sum()
                top_k_images[match] = score

            # Get the top k images by score
            top_images = sorted(top_k_images.keys(), key=lambda image: top_k_images[image], reverse=True)[:k]
            
            # Plotting reranked matches
            plt.figure(figsize=(10, 2))
            for i, img_idx in enumerate(top_images):
                plt.subplot(1, len(top_images), i + 1)
                plt.imshow(images[img_idx].permute(1, 2, 0))  # Assuming images are in CxHxW format
                if img_idx == text_to_image_map[caption_idx].item():
                    plt.title(f"Correct: {img_idx}")
                else:
                    plt.title(f"Reranked: {img_idx}")
                plt.axis('off')
            plt.savefig(f"reranked_matches_caption_{caption_idx}.png")
            plt.close()

            # Check if the correct image is in the top k images
            correct_image = text_to_image_map[caption_idx].item()
            if correct_image in top_images:
                correct_recall_count += 1

        # Compute recall for this k
        print(correct_recall_count)
        recall = correct_recall_count / num_text
        print(recall)
        text_to_image_recall.append(recall)
        
    return text_to_image_recall 
