#!/bin/bash

#SBATCH -p nvidia
#use dalma v100
#SBATCH --gres=gpu:1
#use dalma A100
##SBATCH --gres=gpu:a100:1
#SBATCH -n 10
#SBATCH -t 48:00:00
##SBATCH  --mem=200GB

# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

module purge
source ~/.bashrc
source activate pytorch-1.11.0

#python ./cycleganduiqi.py
python ./cyclegan256.py
#python ./cycleganwuhan256.py
