#!/bin/bash

#SBATCH --account=liupan
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Load
module load nvidia/cuda/9.0
module load google/tensorflow/python3.6-gpu

# Execute python
python ResNet_o.py