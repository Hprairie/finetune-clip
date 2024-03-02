import open_clip
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

original_images = []
images = []

for filename in [filename for filename in os.listdir('images/') if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]

    image = Image.open(os.path.join('images/', filename)).convert("RGB")

    original_images.append(image)
    images.append(preprocess(image))

image_input = torch.tensor(np.stack(images))

# Pooled Embedding similarity
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image_input)

    # Get Pairwise Similarity
    image_features = image_features / image_features.norm(dim=-1)[:, None]
    pair_sim = torch.mm(image_features, image_features.T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pair_sim)
    fig.colorbar(cax)
    plt.savefig('Image_Embedding_Similarity')


# Attention layer similarity for a given embedding
image_input = preprocess(Image.open("images/cat.png")).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features, tokens = model.encode_image(image_input, return_tokens=True)

    # Print tokens
    tokens = torch.squeeze(tokens) 

    # Get Pairwise Similarity
    tokens = tokens / tokens.norm(dim=-1)[:, None]
    pair_sim = torch.mm(tokens, tokens.T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pair_sim)
    fig.colorbar(cax)
    plt.savefig('Image_Token_similarity')


