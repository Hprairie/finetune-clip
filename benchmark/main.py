import sys
import os
from params import parse_args
from open_clip import create_model_and_transforms, get_tokenizer
from dataset import get_dataset, encode_dataset
from retrieval import recall_at_k


def main(args):
    # Get model and transforms
    print("Creating Model...")
    model, _, preprocess = create_model_and_transforms(
            model_name=args.model,
            pretrained=args.pretrained,
            finetune_args=None,
            finetune_path=args.finetune_path
    )
    tokenizer = get_tokenizer(args.model)

    # Setup saving
    print("Creating Savefile...")
    save_path = os.path.join('benchmark-results',args.name)
    os.makedirs(save_path, exist_ok=True)

    # Get Dataset and clean
    print("Fetching Dataset...")
    dataset = get_dataset(args, preprocess, tokenizer)

    # Encode Dataset
    print("Encoding Dataset...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map = encode_dataset(model, dataset, batch_size=args.batchsize)

    # Run Retrieval Benchmark
    print("Running Benchmark...")
    results = recall_at_k(
            image_encodings,
            text_encodings,
            text_to_image_map,
            image_to_text_map,
            patch_to_image_map,
            text_to_encoding_map,
            args.k,
            args.batchsize
    )

    # Save results
    print(results)
    with open(os.path.join(save_path, 'colbert-retrieval.txt'), 'xa') as file:
        for k, val in zip(args.k, results):
            file.write(f'Retrieval @ {k}: {val}')

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)

