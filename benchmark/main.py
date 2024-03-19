import sys
import os
import torch
from params import parse_args
from open_clip import create_model_and_transforms, get_tokenizer
from dataset import get_dataset, encode_dataset
from retrieval import recall_at_k


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get model and transforms
    print("Creating Model...")


    if args.finetune_path is not None:
        model, _, preprocess = create_model_and_transforms(
            args.finetune_args.model,
            '',
            precision=args.finetune_args.precision,
            device=device,
            jit=args.finetune_args.torchscript,
            force_quick_gelu=args.finetune_args.force_quick_gelu,
            force_custom_text=args.finetune_args.force_custom_text,
            force_patch_dropout=args.finetune_args.force_patch_dropout,
            force_image_size=args.finetune_args.force_image_size,
            image_mean=args.finetune_args.image_mean,
            image_std=args.finetune_args.image_std,
            image_interpolation=args.finetune_args.image_interpolation,
            image_resize_mode=args.finetune_args.image_resize_mode,  # only effective for inference
            aug_cfg=args.finetune_args.aug_cfg,
            pretrained_image=args.finetune_args.pretrained_image,
            finetune_args=args.finetune_args,
            finetune_path=args.pretrained
        )
        tokenizer = get_tokenizer(args.finetune_args.model)
    else:
        model, _, preprocess = create_model_and_transforms(
                model_name=args.model,
                pretrained=args.pretrained,
                finetune_args=args.finetune_args,
                finetune_path=args.finetune_path
        )
        tokenizer = get_tokenizer(args.model)

    # Setup saving
    print("Creating Savefile...")
    save_path = os.path.join('benchmark-results',args.name)
    if os.path.isdir(save_path):
        raise FileExistsError("Benchmark directory already exist. Don't overwrite your data :(")
    os.makedirs(save_path)

    # Saving Hyperparameters
    with open(os.path.join(save_path, 'params.txt'), 'w') as file:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            file.write(f"{name}: {val}")

    # Get Dataset and clean
    print("Fetching Dataset...")
    dataset = get_dataset(args, preprocess, tokenizer)

    # Encode Dataset
    print("Encoding Dataset...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map = encode_dataset(model, dataset, batch_size=args.batchsize, reg_retrieval=args.reg_retrieval)

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
            args.batchsize,
            args.reg_retrieval
    )

    # Save results
    print(results)
    with open(os.path.join(save_path, 'colbert-retrieval.txt'), 'w') as file:
        for k, val in zip(args.k, results):
            file.write(f'Retrieval @ {k}: {val}')

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)

