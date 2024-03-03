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
#SBATCH -t 12:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate clip 

# Launch Job
export CUDA_VISIBLE_DEVICES='0'
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd $WORK/projects/retrieval-clip
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
    --name "ViT-B-32-LoRA-Rank32" \
    --pretrained "openai" \
    --lora "32:1" \
    --report-to "wandb" \
    --wandb-project-name "ViT-B-32-Finetune" \
    --log-every-n-steps 100

# ---------------------------------------------------

