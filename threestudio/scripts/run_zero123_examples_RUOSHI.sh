# zero123
python launch.py --config configs/zero123.yaml --train --gpu 0 data.image_path=./load/images/minion_rgba.png system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-zero123xl" system.freq.guidance_eval=13

# zero123XL
python launch.py --config configs/zero123.yaml --train --gpu 0 data.image_path=./load/images/minion_rgba.png system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-zero123xl" system.freq.guidance_eval=13 system.guidance.pretrained_model_name_or_path="/admin/home-vikram/zero123xl/20230605_last.ckpt" tag='XL2_${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}'
