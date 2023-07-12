#!/bin/bash
#SBATCH --job-name=vikky
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:07:00
conda activate three
cd /admin/home-vikram/git/threestudio/
NAME="alien1"
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_DEMO tag="XL_Phase1" system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-zero123XL-demo-NEW2" system.loggers.wandb.name=${NAME}_XL_Phase1
