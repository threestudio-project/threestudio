# Reconstruct Anya using latest Zero123XL, in <2000 steps.
debugpy-run launch.py -- --config configs/zero123.yaml --train system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new" \
 system.loggers.wandb.name="claforte_params" data.image_path=./load/images/anya_front_rgba.png \
 system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt"

# claforte: at first sight, quality plateaued around step ~750
#srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya
python launch.py --config configs/zero123.yaml --train system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new" \
 system.loggers.wandb.name="1000steps" data.image_path=./load/images/anya_front_rgba.png \
 system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt"

# PHASE 2
#srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="baseline" data.random_camera.progressive_until=500
