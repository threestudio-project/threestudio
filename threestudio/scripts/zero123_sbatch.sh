#!/bin/bash
#SBATCH --job-name=vikky_alien1
#SBATCH --account=mod3d
#SBATCH --partition=g40
#SBATCH --gpus=1
#SBATCH --time=0-00:07:00
conda activate three
cd ~/git/threestudio/
NAME=alien1
python launch.py --config configs/experimental/imagecondition_refine.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt='A simple 3D render of an alien' system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt tag=${NAME}_Phase2_refine # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase2
