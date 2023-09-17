#!/bin/bash
#SBATCH --job-name=vikky_cr_ellie
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:30:00
#SBATCH --time=0-00:30:00
source  ~/miniconda3/etc/profile.d/conda.sh
conda activate three
cd ~/git/threestudio/
NAME=cr_ellie
CONFIG=sai_multinoise_amb_fast
MODEL=comb1
ELEV=5.0
AZIM=70.0
PROJECT=AMB
TAG=mn14_AMB_fast
WANDB=True
python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=${PROJECT}/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_${AZIM}azim_${TAG} data.default_elevation_deg=${ELEV} data.default_azimuth_deg=${AZIM} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev_${AZIM}azim_${TAG}
