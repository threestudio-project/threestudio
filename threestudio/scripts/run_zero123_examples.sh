python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-loss-weighted"
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-loss-weighted" \
    data.image_path=./load/images/anya_front_rgba.png 