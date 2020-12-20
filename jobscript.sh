#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J coinrun_eps=01
#BSUB -n 1
#BSUB -W 24:00
#BSUB -u email@email.com
#BSUB -B
#BSUB -N
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Running script..."

module load python3/3.8.0
# module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

python3 src/train.py

