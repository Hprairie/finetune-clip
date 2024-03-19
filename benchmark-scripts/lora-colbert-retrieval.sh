python benchmark/main.py \
    --dataset "$DATASETS/coco/images/val2017" \
    --dataset-ann "$DATASETS/coco/annotations/captions_val2017.json" \
    --pretrained "logs/ViT-B-32-LoRA-Rank4-SPARC/checkpoints/epoch_1.pt" \
    --finetune-path "logs/ViT-B-32-LoRA-Rank4-SPARC" \
    --name "LoRA-Rank4-SPARC-Finegrain" \
    --k 1 5 \
    --batchsize 256

