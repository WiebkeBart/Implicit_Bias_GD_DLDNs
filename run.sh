#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G   # total 32G
#SBATCH --array=0-11%12   # 12 Jobs ## max 12 simultaneously
#SBATCH --output=slurm-%x_%A_%a.out
#SBATCH --error=error-%x_%A_%a.err

N=100
M=20
s=3
L=3
SEED=23

uv run --locked python -m gradient_descent.main $SLURM_ARRAY_TASK_ID \
    --seed $SEED --N $N --M $M --s $s --L $L\
    --output_dir "job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
