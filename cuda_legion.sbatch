#!/bin/bash
#SBATCH --time=230:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=trainingCycleGAN
#SBATCH --mem=48GB
#SBATCH --mail-type=ALL
##
# load cuda module
module load nvidia/cudasdk/10.0

python train.py