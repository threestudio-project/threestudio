#!/bin/bash
#SBATCH --job-name=vikky_yoga3
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:30:00
conda activate three
cd ~/git/threestudio/
NAME=yoga3
CONFIG=3sai
MODEL=elevcond2
ELEV=5.0
PROJECT=Zero123_Dmitry_comp
TAG=
WANDB=true
python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=${PROJECT}/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_${TAG} data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev_${TAG}
