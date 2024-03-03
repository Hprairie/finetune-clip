import hnswlib
import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import collections


def recall_at_k(
        image_encodings,
        text_encodings,
        text_to_image_map,
        image_to_text_map,
        patch_to_image_map,
        text_to_encoding_map,
        k_vals,
        batch_size
    ):
    
    image_encodings = torch.cat([patches for patches in image_encodings], dim=0)
    text_encodings = torch.cat([text for text in text_encodings], dim=0)
 
    num_text = len(text_to_image_map)
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    
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
    n_patches = 50

    for k in k_vals:
        correct_recall_count = 0
        
        for caption_idx in range(num_text):
            token_to_image_scores = {}
            
            # Get the text encodings for this caption
            caption_text_encodings = text_encodings[caption_idx * 77 : (caption_idx + 1) * 77]
            
            # Get the position of the EOT token which is the highest number in the sequence
            eot_idx = torch.argmax(caption_text_encodings[:, -1]) # <---- One thing to note is that they use the extra padding tokens 
                                                                  #       in the ColBert paper and it help performance, would be good
                                                                  #       to test with and without it.
    
            assert eot_idx <= 77 
            
            # print(eot_idx)

            # Image's that we want to run full maxsim
            image_matches = set()
            
            # For each token in the caption until eot_idx
            for token_idx in range(caption_idx * 77, caption_idx * 77 + eot_idx):
            # for token_idx in range(caption_idx * 77, (caption_idx + 1) * 77):
                
                # Query the kNN for n_patches closest patches
                labels, distances = p.knn_query(text_encodings[token_idx].cpu().unsqueeze(0), k=n_patches)
                labels, distances = labels[0], distances[0]

                # Map each path to it's parent and add it to set
                for label in labels:
                    parent_image = patch_to_image_map[label].item()

                    # Add the parent_image to the set of working images
                    image_matches.add(parent_image)
                    
            # For all found image's calculate the similarity scores
            import pdb; pdb.set_trace()
            top_k_images = collections.defaultdict(float)
            for match in range(len(image_matches)): # <- Parallelize this and increase batch_size
                # Get patch embeddings
                patches = image_encodings[image_matches[match]]
                
                # Calculate Cosine similarity
                scaled_text_embeddings = caption_text_encodings / torch.norm(caption_text_encodings, dim=1)
                scaled_image_embeddings = patches / torch.norm(patches, dim=1)
                cos_sim  = scaled_text_embeddings @ scaled_image_embeddings.T

                # Max pool per embedding
                score = sum(torch.argmax(cos_sim, dim=0))
                top_k_images[image_matches[match]] = score

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

