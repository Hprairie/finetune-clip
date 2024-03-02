export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

torchrun --nproc_per_node 1 -m finetune.main \
    --dataset-type "csv" \
    --train-data "$DATASETS/coco/annotations/captions_train2017.csv" \
    --warmup 1000 \
    --batch-size 256 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 2 \
    --model "ViT-B-32" \
    --name "ViT-B-32-LoRA-Rank4" \
    --pretrained "openai" \
    --lora "4:1" \
    --report-to "wandb" \
    --wandb-project-name "ViT-B-32-LoRA" \
    --log-every-n-steps 100
