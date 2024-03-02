#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J FT-CLIP                        # Job name
#SBATCH -o FT-CLIP.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e FT-CLIP.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100                       # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 3                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=haydenprairie@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate clip 

# Launch Job
export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd $WORK/projects/retrival-clip
srun --cpu_bind=v --accel-bind=gn python -m finetune.main \
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
    --wanb-project-name "ViT-B-32-LoRA" \
    --log-every-n-steps 100

# ---------------------------------------------------

