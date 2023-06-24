# Reconstruct Anya using latest Zero123XL, in <2000 steps.
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-anya-new" system.loggers.wandb.name="claforte_params" data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 system.prompt_processor.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt"

# PHASE 2
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train --gpu 6 \
  system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )"\
  system.weights=outputs/zero123/64_anya_front_rgba.png@20230622-203756/ckpts/last.ckpt system.freq.guidance_eval=13 \
  system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-phase2" \
  data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="anya21_lambda_3d_normal_smooth=0.7" data.random_camera.progressive_until=500
