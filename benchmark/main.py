import sys
import os
import torch
from params import parse_args
from open_clip import create_model_and_transforms, get_tokenizer
from dataset import get_dataset, encode_dataset
from retrieval import recall_at_k, reranker_recall_at_k
import logging

def get_finetuned_model(device, pretrained, finetune_args):
    return create_model_and_transforms(
            finetune_args.model,
            '',
            precision=finetune_args.precision,
            device=device,
            jit=finetune_args.torchscript,
            force_quick_gelu=finetune_args.force_quick_gelu,
            force_custom_text=finetune_args.force_custom_text,
            force_patch_dropout=finetune_args.force_patch_dropout,
            force_image_size=finetune_args.force_image_size,
            image_mean=finetune_args.image_mean,
            image_std=finetune_args.image_std,
            image_interpolation=finetune_args.image_interpolation,
            image_resize_mode=finetune_args.image_resize_mode,  # only effective for inference
            aug_cfg=finetune_args.aug_cfg,
            pretrained_image=finetune_args.pretrained_image,
            context_length=args.context_length,
            finetune_args=finetune_args,
            finetune_path=pretrained
        )

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get model and transforms
    logging.info("Creating Model...")


    if args.finetune_path is not None:
        model, _, preprocess = get_finetuned_model(device, args.pretrained, args.finetune_args)
        tokenizer = get_tokenizer(args.finetune_args.model, context_length=args.context_length)
    else:
        model, _, preprocess = create_model_and_transforms(
                model_name=args.model,
                pretrained=args.pretrained,
        )
        tokenizer = get_tokenizer(args.model, args.context_length)

    reranker_model = None
    if args.reranker_finetune_path is not None:
        reranker_model, _, _ = get_finetuned_model(device, args.pretrained_reranker, args.reranker_args)
    elif args.reranker_model is not None:
        reranker_model, _, _ = create_model_and_transforms(
                model_name=args.reranker_model,
                pretrained=args.pretrained_reranker,
        )

    # Setup saving
    logging.info("Creating Savefile...")
    save_path = os.path.join('benchmark-results',args.name)
    if os.path.isdir(save_path):
        raise FileExistsError("Benchmark directory already exist. Don't overwrite your data :(")
    os.makedirs(save_path)

    # Saving Hyperparameters
    with open(os.path.join(save_path, 'params.txt'), 'w') as file:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            file.write(f"{name}: {val}\n")

    # Get Dataset and clean
    logging.info("Fetching Dataset...")
    dataset = get_dataset(args, preprocess, tokenizer, mask_padding=args.mask_padding, repeat_tokens=args.repeat_tokens)

    # Encode Dataset
    logging.info("Encoding Dataset...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map, patch_to_image_map, text_to_encoding_map, masks = encode_dataset(model, dataset, batch_size=args.batchsize, reg_retrieval=args.reg_retrieval, mask_padding=args.mask_padding)

    # Run Retrieval Benchmark
    logging.info("Running Benchmark...")
    if reranker_model is not None:
        reranker_image_encodings, reranker_text_encodings, text_to_image_map, image_to_text_map, _, _, _ = encode_dataset(reranker_model, dataset, batch_size=args.batchsize, reg_retrieval=False, second_to_last=args.second_to_last)
        
        results = reranker_recall_at_k(
                image_encodings,
                text_encodings,
                reranker_image_encodings,
                reranker_text_encodings,
                text_to_image_map,
                image_to_text_map,
                args.k,
                args.batchsize,
                args.reranker_multiple
        )
    else:
        results = recall_at_k(
                image_encodings,
                text_encodings,
                text_to_image_map,
                image_to_text_map,
                patch_to_image_map,
                text_to_encoding_map,
                masks,
                args.k,
                args.batchsize,
                args.reg_retrieval
        )

    # Save results
    logging.info(results)
    with open(os.path.join(save_path, 'retrieval-results.txt'), 'w') as file:
        for k, val in zip(args.k, results):
            file.write(f'Retrieval @ {k}: {val}\n')

if __name__ == '__main__':
    args = sys.argv[1:]
    args = parse_args(args)

    main(args)

