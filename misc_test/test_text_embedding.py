import open_clip
import torch
from PIL import Image
import matplotlib.pyplot as plt


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

text = tokenizer(["a photo of an orange cat and a blue dog"])

# Attention layer similarity for a given embedding
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features, tokens = model.encode_text(text, return_tokens=True)

    # Print tokens
    tokens = torch.squeeze(tokens) 
    tokens = tokens[:13]

    # Get Pairwise Similarity
    tokens = tokens / tokens.norm(dim=-1)[:, None]
    pair_sim = torch.mm(tokens, tokens.T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pair_sim)
    fig.colorbar(cax)
    plt.savefig('Text_Token_similarity')

# Pooled Embedding similarity
text = tokenizer(["A large monkey", "brown cat", "eiffel tower", "blue sky", "car that is stripped", "--!0% kat"])
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text)

    # Get Pairwise Similarity
    text_features = text_features / text_features.norm(dim=-1)[:, None]
    pair_sim = torch.mm(text_features, text_features.T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pair_sim)
    fig.colorbar(cax)
    plt.savefig('Text_Embedding_Similarity')
