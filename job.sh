#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH -t 23:00:00
#SBATCH --mem 64G
#SBATCH --job-name="Generating synthetic data"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kjmetzler@wpi.edu

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name

python models/run_generator.py
