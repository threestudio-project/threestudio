python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-anya-new" system.loggers.wandb.name="dragon2" data.image_path=./load/images/dragon2_rgba.png system.freq.guidance_eval=0 system.prompt_processor.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt"

python threestudio/scripts/make_training_vid.py --exp /admin/home-vikram/git/threestudio/outputs/zero123/64_dragon2_rgba.png@20230628-152734 --frames_per_vid 30 --fps 20 --max_iters 200
