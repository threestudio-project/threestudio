NAME="dragon2"

# Phase 1 - 64x64
python launch.py --config configs/zero123.yaml --train --gpu 7 data.image_path=./load/images/${NAME}_rgba.png use_timestamp=False name=${NAME} tag=Phase1 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase1

# Phase 1.5 - 512 refine
python launch.py --config configs/zero123-geometry.yaml --train --gpu 4 data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase1p5

# Phase 2 - dreamfusion
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train --gpu 5 data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a friendly dragon" system.weights="/admin/home-vikram/git/threestudio/outputs/${NAME}/Phase1/ckpts/last.ckpt" name=${NAME} tag=Phase2 # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase2

# Phase 2 - SDF + dreamfusion
python launch.py --config configs/experimental/imagecondition_zero123nerf_refine.yaml --train --gpu 5 data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a friendly dragon" system.geometry_convert_from="/admin/home-vikram/git/threestudio/outputs/${NAME}/Phase1/ckpts/last.ckpt" name=${NAME} tag=Phase2_refine # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project="zero123" system.loggers.wandb.name=${NAME}_Phase2_refine
