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
module load nvidia/cudasdk/10.1
module load singularity/3.2.1

singularity exec -s /bin/bash --nv maio9.img bash -c 'python ./CycleNLPGAN/train.py --name translation --dataroot data --batch_size 8 --task translation'