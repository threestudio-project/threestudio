NAME="dragon2"

# Phase 1 - 64x64
python launch.py --config configs/zero123_64.yaml --train --gpu 7 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-anya-new" system.loggers.wandb.name=${NAME} data.image_path=./load/images/${NAME}_rgba.png system.freq.guidance_eval=0 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME} tag="Phase1_64"

# python threestudio/scripts/make_training_vid.py --exp /admin/home-vikram/git/threestudio/outputs/zero123/64_dragon2_rgba.png@20230628-152734 --frames_per_vid 30 --fps 20 --max_iters 200

# # Phase 1.5 - 512
# python launch.py --config configs/zero123_512.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-zero123XL-demo" system.loggers.wandb.name="robot_512_drel_n_XL_SAMEgeom" data.image_path=./load/images/robot_rgba.png system.freq.guidance_eval=0 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" tag='${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}_XL_SAMEgeom' system.weights="/admin/home-vikram/git/threestudio/outputs/zero123/[64, 128]_robot_rgba.png_OLD@20230630-052314/ckpts/last.ckpt"

# Phase 1.5 - 512 refine
python launch.py --config configs/zero123-geometry.yaml --train --gpu 4 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-zero123XL-demo" system.loggers.wandb.name="robot_512_drel_n_XL_SAMEg" system.freq.guidance_eval=0 data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1_64/ckpts/last.ckpt use_timestamp=False name=${NAME} tag="Phase2_512geom"

# Phase 2 - dreamfusion
python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train --gpu 5 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-zero123XL-demo" system.loggers.wandb.name="robot_512_drel_n_XL_SAMEw" tag='${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}_XL_Phase2' system.freq.guidance_eval=0 data.image_path=./load/images/robot_rgba.png system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" system.weights="/admin/home-vikram/git/threestudio/outputs/zero123/[64, 128]_robot_rgba.png_OLD@20230630-052314/ckpts/last.ckpt"

python launch.py --config configs/experimental/imagecondition_zero123nerf_refine.yaml --train --gpu 5 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-zero123XL-demo" system.loggers.wandb.name="robot_512_drel_n_XL_SAMEw" tag='${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}_XL_Phase2_refine' system.freq.guidance_eval=0 data.image_path=./load/images/robot_rgba.png system.prompt_processor.prompt="A 3D model of a friendly dragon" system.geometry_convert_from="/admin/home-vikram/git/threestudio/outputs/zero123/[64, 128, 256]_dragon2_rgba.png_XL_REPEAT@20230705-023531/ckpts/last.ckpt"

# A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )"
# "/admin/home-vikram/git/threestudio/outputs/zero123/[64, 128]_robot_rgba.png_OLD@20230630-052314/ckpts/last.ckpt"

# Adds zero123_512-refine.yaml
# Adds resolution_milestones to image.py
# guidance_eval gets max batch_size 4
# Introduces random_bg in solid_color_bg
