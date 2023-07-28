#!/bin/bash
#SBATCH --job-name=vikky_temple2
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:15:00
conda activate three
cd ~/git/threestudio/
NAME=temple2
TYPE=sai
python launch.py --config configs/zero123_${TYPE}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png use_timestamp=false name=SAI/${NAME} tag=Phase1_${TYPE}_EXPconst0.5 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123_SAI' system.loggers.wandb.name=${NAME}_${TYPE}
