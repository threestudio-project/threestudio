
GPU_ID=$1         # e.g. 0
IMAGE_PREFIX=$2   # e.g. "anya_front"
ZERO123_PREFIX=$3 # e.g. "XL_20230604"
ELEVATION=$4      # e.g. 0
REST=${@:5:99}    # e.g. "system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9"

# change this config if you don't use wandb or want to speed up training
python launch.py --config configs/zero123.yaml --train --gpu $GPU_ID system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-noise_atten" \
    system.loggers.wandb.name="${IMAGE_PREFIX}_zero123_${ZERO123_PREFIX}...fov20_${REST}" \
    data.image_path=./load/images/${IMAGE_PREFIX}_rgba.png system.freq.guidance_eval=37 \
    system.guidance.pretrained_model_name_or_path="./load/zero123/${ZERO123_PREFIX}.ckpt" \
    system.guidance.cond_elevation_deg=$ELEVATION \
    ${REST}
