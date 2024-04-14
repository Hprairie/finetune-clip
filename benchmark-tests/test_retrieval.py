import torch
import sys
from scipy.stats import unitary_group

sys.path.append('./benchmark')
from retrieval import finegrained_recall_at_k


def test_finegrain_recall_at_k_old(k, dim, samples):
    image_encodings: list = [torch.tensor(unitary_group.rvs(dim)) for _ in range(samples)]
    text_encodings: np.array = [torch.conj_physical(matrix).T for matrix in image_encodings] 
    text_to_image_map: np.array = torch.tensor([[i] for i in range(samples)])
    image_to_text_map: np.array = torch.tensor([[i] for i in range(samples)])
    patch_to_image_map: np.array = torch.tensor([[i] * dim for i in range(samples)])
    text_to_encoding_map: None = None
    batch_size: None = None

    # Test Finegrain Recall
    results = finegrained_recall_at_k(
        image_encodings=image_encodings,
        text_encodings=text_encodings,
        text_to_image_map=text_to_image_map,
        image_to_text_map=image_to_text_map,
        patch_to_image_map=patch_to_image_map,
        text_to_encoding_map=text_to_encoding_map,
        k_vals=[k],
        batch_size=batch_size
    )

    print(results)
    
def test_finegrain_recall_at_k(k, tokens, patches, dim, samples):
    if tokens > patches:
        text_encodings: list = [torch.rand(tokens, dim) for _ in range(samples)]
        image_encodings: list = [matrix[:patches] for matrix in text_encodings] 
    else:
        image_encodings: list = [torch.rand(patches, dim) for _ in range(samples)]
        text_encodings: list = [matrix[:tokens] for matrix in image_encodings] 
    text_to_image_map: torch.Tensor = torch.tensor([[i] for i in range(samples)])
    image_to_text_map: np.array = torch.tensor([[i] for i in range(samples)])
    patch_to_image_map: np.array = torch.tensor([[i] * patches for i in range(samples)]).flatten()
    text_to_encoding_map: None = None
    batch_size: None = None

    # Test Finegrain Recall
    results = finegrained_recall_at_k(
        image_encodings=image_encodings,
        text_encodings=text_encodings,
        text_to_image_map=text_to_image_map,
        image_to_text_map=image_to_text_map,
        patch_to_image_map=patch_to_image_map,
        text_to_encoding_map=text_to_encoding_map,
        k_vals=k,
        batch_size=batch_size
    )

    print(results)


if __name__ == "__main__":
    test_finegrain_recall_at_k(k=[1,5,10,25], tokens=77, patches=49, dim=512, samples=8096)
