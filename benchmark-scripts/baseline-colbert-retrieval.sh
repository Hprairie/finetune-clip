python benchmark/main.py \
    --dataset "$DATASETS/coco/images/train2017" \
    --dataset-ann "$DATASETS/coco/annotations/captions_train2017.json" \
    --model "ViT-B-32" \
    --pretrained "openai" \
    --name "ViT-B-32-Base" \
    --k 1 5 \
    --batchsize 256

