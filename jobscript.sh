#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J ValueClip_test
#BSUB -n 1
#BSUB -W 10:00
#BSUB -u tdheshe@hotmail.com
#BSUB -B
â#BSUB -N
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
echo "Running script..."

module load python3/3.8.0
# module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

python3 encoder_baseline_ppo.py

