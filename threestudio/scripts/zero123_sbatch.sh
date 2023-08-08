#!/bin/bash
#SBATCH --job-name=vikky_yoga3
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:15:00
conda activate three
cd ~/git/threestudio/
NAME=yoga3
CONFIG=sai
MODEL=sai
ELEV=5.0
python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=SAI/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_sp0.1to0.5 data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=false system.loggers.wandb.project='zero123_SAIconfig_comp' system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev
