#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J FT-CLIP                        # Job name
#SBATCH -o slurmlogs/FT-CLIP.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/FT-CLIP.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 20:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=anshita@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate base 

# Launch Job
export CUDA_VISIBLE_DEVICES='0'
export MASTER_PORT=12802
w
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd $WORK/research/finetune-clip/open_clip/src
srun --cpu_bind=v --accel-bind=gn python -m finetune.main \
    --dataset-type "csv" \
    --train-data "$WORK/research/MSCOCO/captions_train2017.csv" \
    --warmup 1000 \
    --batch-size 128 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 2 \
    --model "ViT-B-32" \
    --name "ViT-B-32-LoRA-Rank4-ColBERT-CC3M-MSCOCO" \
    --pretrained "$WORK/research/finetune-clip/open_clip/src/logs/ViT-B-32-LoRA-Rank4-ColBERT-CC3M/checkpoints/epoch_1.pt" \
    --finetune-path "$WORK/research/finetune-clip/open_clip/src/logs/ViT-B-32-LoRA-Rank4-ColBERT-CC3M" \
    --lora "4:1" \
    --colbert-local-contrastive "token-wise" \
    --colbert-global-contrastive "text-wise" \
    --colbert \
    --report-to "wandb" \
    --wandb-project-name "ViT-B-32-Finetune" \
    --log-every-n-steps 100

# ---------------------------------------------------

