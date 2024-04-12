#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J FT-CLIP                        # Job name
#SBATCH -o slurmlogs/BM-CLIP.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/BM-CLIP.e%j          # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=anshita@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A MLL                            # Allocation name


# Source conda environment
source $WORK/miniconda3/bin/activate base 

# Launch Job

cd $WORK/research/finetune-clip/
python benchmark/main.py \
    --dataset "/scratch/09765/anshita/MSCOCO/val2017" \
    --dataset-ann "/scratch/09765/anshita/MSCOCO/captions_val2017.json" \
    --model "ViT-B-32" \
    --pretrained "openai" \
    --pretrained-reranker "$WORK/research/finetune-clip/open_clip/src/logs/ViT-B-32-LoRA-Epoch1-Rank4-ColBERT-CC3M/checkpoints/epoch_1.pt" \
    --reranker-finetune-path "$WORK/research/finetune-clip/open_clip/src/logs/ViT-B-32-LoRA-Epoch1-Rank4-ColBERT-CC3M" \
    --name "Base-With-Reranker-LoRA-CC3M-FG-5k" \
    --k 1 5 10 \
    --batchsize 128 \
    --reg-retrieval \

# ---------------------------------------------------
