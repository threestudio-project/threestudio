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

# test7_lambda_sparsity=4.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test7_lambda_sparsity=4.0"

# test8_lambda_opacity_smooth=1.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test8_lambda_opacity_smooth=1.0"

# test9_lambda_opacity_smooth=100.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test9_lambda_opacity_smooth=100.0"

# test10_lambda_opacity_smooth=20.0;lambda_depth_smooth=5
# lambda_depth_smooth: 5.0
# lambda_opacity_smooth: 20.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test10_lambda_opacity_smooth=20.0;lambda_depth_smooth=5"

# test11_lambda_depth_smooth=50
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test11_lambda_depth_smooth=50"

# test12_abs_total_variation
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test12_abs_total_variation"

# test13_lambda_rgb_smooth=5.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test13_lambda_rgb_smooth=5.0"

# test14_lr=0.05_10X
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test14_lr=0.05_10X"

# test15_lr=0.15_3X
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test15_lr=0.15_3X"

# test16_progressive_until=0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test16_progressive_until=0"

# test17_min_step_percent=[0,0.1,0.02,2000];max_step_percent:[0,0.45,0.45,2000]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test17_min_step_percent=[0,0.1,0.02,2000];max_step_percent:[0,0.45,0.45,2000]"

# test18_min_step_percent=[0,0.1,0.02,2000];max_step_percent:[0,0.7,0.5,2000]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test18_min_step_percent=[0,0.1,0.02,2000];max_step_percent:[0,0.7,0.5,2000]"

# test19_min_step_percent=[0,0.03,0.02,2000];max_step_percent:[0,0.3,0.3,2000]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/zero123/[64, 128]_anya_front_rgba.png_prog0@20230706-183840/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test19_min_step_percent=[0,0.03,0.02,2000];max_step_percent:[0,0.3,0.3,2000]"

# test20_use_Vikram_anya
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test20_use_Vikram_anya"

# test21_use_Vikram_anya2_correct_image
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test21_use_Vikram_anya2_correct_image"

# test22_use_Vikram_anya3_finally_correct_image
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test22_use_Vikram_anya3_finally_correct_image"

# test23_res=256_bs=6
# height: 256
# width: 256
# batch_size: 6
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test23_res"

# test24_1000steps_changed_fov
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test24_1000steps_changed_fov"

# test25_min_step_percent:[0,0.02,0.01,2000];max_step_percent:[0,0.15,0.15,2000]
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test25_min_step_percent:[0,0.02,0.01,2000];max_step_percent:[0,0.15,0.15,2000]"


# test26_min_step_percent:[0,0.4,0.2,200];max_step_percent:[0,0.85,0.5,200];
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test26_min_step_percent:[0,0.4,0.2,200];max_step_percent:[0,0.85,0.5,200]"

# test27_lambda_sds=5.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test27_lambda_sds=5.0"

# test28_lambda_sds=10.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test28_lambda_sds=10.0"

# test29_lambda_sds=2.0
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test29_lambda_sds=2.0"

# test30_lambda_sds=0.3
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" system.freq.guidance_eval=13 \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="test29_lambda_sds=0.3"
