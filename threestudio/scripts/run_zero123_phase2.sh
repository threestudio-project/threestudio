# Reconstruct Anya using latest Zero123XL, in <2000 steps.
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-anya-new" system.loggers.wandb.name="claforte_params" data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt"

# PHASE 2
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train --gpu 0 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" system.weights=outputs/zero123/128_anya_front_rgba.png@20230623-145711/ckpts/last.ckpt system.freq.guidance_eval=13 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-anya-new" data.image_path=./load/images/anya_front_rgba.png system.loggers.wandb.name="anya" data.random_camera.progressive_until=500

# test101_main_default_deepfloyd_with_hashgrid
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/dreamfusion-if.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 system.loggers.wandb.name="test101_main_default_deepfloyd_with_hashgrid"

# test102_main_default_sd_with_hashgrid
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/dreamfusion-sd.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 system.loggers.wandb.name="test102_main_default_sd_with_hashgrid"

# test103_main_default_sd_albedo_activation=scale_-11_01
# otherwise, like test102_main_default_sd_with_hashgrid
# results: outputs/dreamfusion-sd/A_DSLR_3D_photo_of_a_cute_anime_schoolgirl_stands_proudly_with_her_arms_in_the_air,_pink_hair_(_unreal_engine_5_trending_on_Artstation_Ghibli_4k_)@20230710-170705/save
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/dreamfusion-sd.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 system.loggers.wandb.name="test103_main_default_sd_albedo_activation=scale_-11_01"

# test104_main_default_sd_albedo_activation=scale_-11_01;guidance_scale=20
# otherwise, like test102_main_default_sd_with_hashgrid
# results: outputs/dreamfusion-sd/A_DSLR_3D_photo_of_a_cute_anime_schoolgirl_stands_proudly_with_her_arms_in_the_air,_pink_hair_(_unreal_engine_5_trending_on_Artstation_Ghibli_4k_)@20230710-171146
# Conclusion: unstable once the shading is enabled (diverged around step 1400)
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/dreamfusion-sd.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 system.loggers.wandb.name="test104_main_default_sd_albedo_activation=scale_-11_01;guidance_scale=20"

# test105_grid_prune=false
# otherwise, like test104
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/dreamfusion-sd.yaml --train \
 system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
 system.weights="outputs/anya_front/Phase1/ckpts/last.ckpt" \
 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-new-ph2" \
 system.loggers.wandb.name="test105_grid_prune=false"




# test108_zero123_anneal_density_bias_1000steps
# Phase 1
NAME="anya_front"
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png \
 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" \
 use_timestamp=False name=${NAME} tag="Phase1_annealed_density_bias" \
 system.freq.guidance_eval=37 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-ph1-again" system.loggers.wandb.name="test108_zero123_anneal_density_bias_1000steps"

# test109_zero123_1000steps_batch_size=[12,4,1]
# Phase 1
NAME="anya_front"
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png \
 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" \
 use_timestamp=False name=${NAME} tag="Phase1_annealed_density_bias" \
 system.freq.guidance_eval=37 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-ph1-again" system.loggers.wandb.name="test109_zero123_1000steps_batch_size=[12,4,1]"

# test110_zero123_500steps_lots_of_changes
# Phase 1
NAME="anya_front"
srun --account mod3d --partition=g40 --gpus=1 --job-name=3s_anya \
python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png \
 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" \
 use_timestamp=False name=${NAME} tag="Phase1_annealed_density_bias" \
 system.freq.guidance_eval=37 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-anya-ph1-again" system.loggers.wandb.name="test110_zero123_500steps_lots_of_changes"
