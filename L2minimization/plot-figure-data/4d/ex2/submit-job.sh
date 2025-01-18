#!/bin/bash --login 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH -J greedy4DQMCex1
#SBATCH -o greedy4DQMCex2.%J.out
#SBATCH -e greedy4DQMCex2.%J.err
#SBATCH --time=24:00:00
#SBATCH --mail-user=xiaofeng.xu@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --mem=32G
#run the application:
module load mpich
conda activate torch
export OMP_NUM_THREADS=1
python Greedy4DQMC.py
