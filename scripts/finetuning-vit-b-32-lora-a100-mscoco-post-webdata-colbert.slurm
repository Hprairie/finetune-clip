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
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
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
    --dataset-type "webdataset" \
    --train-data "/scratch/08002/gsmyrnis/cc3m/{00000..00331}.tar" \
    --train-num-samples 2905954 \
    --warmup 1000 \
    --batch-size 128 \
    --lr 1e-6 \
    --wd 0.1 \
    --epochs 1 \
    --workers 2 \
    --model "ViT-B-32" \
    --pretrained "openai" \
    --name "ViT-B-32-LoRA-Epoch1-LR1e-6-Rank8-ColBERT-(TEXT,TOKEN)-CC3M" \
    --lora "8:1" \
    --colbert \
    --colbert-local-contrastive 'token-wise' \
    --colbert-global-contrastive 'text-wise' \
    --report-to "wandb" \
    --wandb-project-name "ViT-B-32-Finetune-CC3M" \
    --log-every-n-steps 100

# ---------------------------------------------------
#    --colbert-dropout 0.1 \

