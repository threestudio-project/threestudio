python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl2" \
    system.loggers.wandb.name="anya_zero123XL_20230604" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# default zero123 ckpt
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl2" \
    system.loggers.wandb.name="anya_zero123_105000" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9
