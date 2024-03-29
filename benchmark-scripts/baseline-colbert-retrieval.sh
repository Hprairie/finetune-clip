python benchmark/main.py \
    --dataset "$DATASETS/coco/images/val2017" \
    --dataset-ann "$DATASETS/coco/annotations/captions_val2017.json" \
    --model "ViT-B-32" \
    --pretrained "openai" \
    --name "ViT-B-32-Baseline" \
    --k 1 5 \
    --batchsize 256
