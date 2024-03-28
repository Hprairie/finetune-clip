source $WORK/miniconda3/bin/activate clip 

# Launch Job
export CUDA_VISIBLE_DEVICES='0'
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun --cpu_bind=v --accel-bind=gn python -m finetune.main \
    --dataset-type "csv" \
    --train-data "$DATASETS/coco/annotations/captions_train2017.csv" \
    --warmup 1000 \
    --batch-size 128 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 2 \
    --model "ViT-B-32" \
    --name "ViT-B-32-LoRA-Epoch1-Rank4-ColBERT-Testing" \
    --lora "4:1" \
    --colbert \
    --colbert-local-contrastive 'all' \
    --colbert-global-contrastive 'all' \
    --report-to "wandb" \
    --wandb-project-name "ViT-B-32-Finetune" \
    --log-every-n-steps 100
