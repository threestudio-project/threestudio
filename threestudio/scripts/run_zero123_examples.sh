debugpy-run launch.py -- --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    system.freq.ref_or_zero123="accumulate" freq.guidance_eval=13 system.freq.guidance_eval=13 trainer.max_steps=5000

python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    system.freq.ref_or_zero123="alternate" freq.guidance_eval=13 system.freq.guidance_eval=13

python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 trainer.max_steps=5000

python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13
