#!/bin/bash
#SBATCH --job-name=vikky_temple2
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:15:00
conda activate three
cd ~/git/threestudio/
NAME=temple2
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png use_timestamp=false name=${NAME} tag=Phase1_highNoise_lr0.001fast_frontAvoid5_norm50 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1
