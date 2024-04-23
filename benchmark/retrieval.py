import hnswlib
import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import logging
import collections

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
        reg_retrieval
    ):

    if reg_retrieval:
        return reg_recall_at_k(
            image_encodings,
            text_encodings,
            text_to_image_map,
            image_to_text_map,
            k_vals,
            batch_size
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
        batch_size
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
        batch_size
    ):
    
    image_encodings = torch.cat([patches for patches in image_encodings], dim=0)
    text_encodings = torch.cat([text for text in text_encodings], dim=0)

    num_text = len(text_to_image_map)
    num_im = image_encodings.shape[0]
    
    print(image_encodings.shape)
    print(text_encodings.shape)
    print(masks.shape)

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
            caption_text_encodings = text_encodings[caption_idx * 77 : (caption_idx + 1) * 77]

            # Images that we want to run full maxsim on
            image_matches = set()
            
            # For each token in the caption excluding padding tokens
            for token_idx in range(caption_idx * 77, (caption_idx + 1) * 77):
                if len(masks) > 0 and masks[caption_idx][token_idx - (caption_idx * 77)] == 0:
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
        k_vals,
        batch_size,
        k_multiple
    ):

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
    text_to_image_recall = []
    
    print("image encodings: ", fg_image_encodings.shape)
    print("text encodings: ", fg_text_encodings.shape)

    for k in k_vals:
        correct_recall_count = 0
        for caption_idx in range(num_text):
            
            labels, distances = p.knn_query(coarse_text_encodings[caption_idx].cpu().numpy(), k=int(k*k_multiple))
            image_matches = set(labels[0])

            # For all found images calculate the similarity scores
            top_k_images = collections.defaultdict(float)
            caption_text_encodings = fg_text_encodings[caption_idx * 77 : (caption_idx + 1) * 77]

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