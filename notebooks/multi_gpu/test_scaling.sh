#!/bin/bash
#SBATCH --job-name=scaling-jax
#SBATCH --account=jf1uids
#SBATCH --output=scaling_jax_%j.out
#SBATCH --error=scaling_jax_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load any necessary modules (optional, depending on your cluster)
# module load cuda/11.8

# allow all GPUs to be used
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate your Conda environment
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate jf1uids

# get gpustats
gpustat

# Run your Python script
python -u scaling_jax.py