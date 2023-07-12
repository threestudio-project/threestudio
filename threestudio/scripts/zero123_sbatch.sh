#!/bin/bash
#SBATCH --job-name=vikky
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:07:00
conda activate three
cd ~/git/threestudio/
NAME="dog1"
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png use_timestamp=False name=${NAME} tag=Phase1 system.loggers.wandb.enable=true system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1
