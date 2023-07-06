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


# test1
# camera_distance_range: [3.5, 3.8]
# fovy_range: [20.0, 25.0] # Zero123 has fixed fovy
# camera_perturb: 0.1
# center_perturb: 0.1
# up_perturb: 0.05
# ...
# min_step_percent: [0, 0.2, 0.02, 2000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.45, 0.3, 2000]
#
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test1" data.random_camera.progressive_until=500

# test2_ambient_ratio=0.7+0.3*rand
# don't favor white bg,
# ambient_ratio = 0.7 + 0.3 * random.random()
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test2_ambient_ratio=0.7+0.3*rand" data.random_camera.progressive_until=500

# test3_progressive_until_1500
# progressive_until: 1500
# camera_perturb: 0.05
# center_perturb: 0.05
# ...
# ambient_only_steps: 100
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test3_progressive_until_1500"

# test4_camera_distance_range=[3.1,3.8]
# camera_distance_range: [3.1, 3.8]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test4_camera_distance_range=[3.1,3.8]"

# test5_min_step_percent=[0,0.1,0.02,2000];max_step_percent=[0,0.3,0.3,2000]
# min_step_percent: [0, 0.1, 0.02, 2000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.3, 0.3, 2000]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test5_min_step_percent=[0,0.1,0.02,2000];max_step_percent=[0,0.3,0.3,2000]"


# test6_lambda_sparsity=0.2
# lambda_sparsity: 0.2
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test6_lambda_sparsity=0.2"
