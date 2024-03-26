#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J BM-CLIP                        # Job name
#SBATCH -o slurmlogs/BM-CLIP.o%j          # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e slurmlogs/BM-CLIP.e%j          # Name of stderr error file (%j corresponds to the job id)
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

cd $WORK/projects/retrieval-clip
python benchmark/main.py \
    --dataset "$DATASETS/coco/images/val2017" \
    --dataset-ann "$DATASETS/coco/annotations/captions_val2017.json" \
    --pretrained "openai" \
    --model "ViT-B-32" \
    --name "Baseline-ViT-B-32-Colbert" \
    --k 1 5 \
    --batchsize 256

# ---------------------------------------------------
