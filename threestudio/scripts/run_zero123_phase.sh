
GPU_ID=$1         # e.g. 0
IMAGE_PREFIX=$2   # e.g. "anya_front"
ZERO123_PREFIX=$3 # e.g. "zero123XL_20230604"
ELEVATION=$4      # e.g. 0

python launch.py --config configs/zero123.yaml --train --gpu $GPU_ID system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl_challenges" \
    system.loggers.wandb.name="${IMAGE_PREFIX}_zero123_${ZERO123_PREFIX}" \
    data.image_path=./load/images/${IMAGE_PREFIX}_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.guidance.pretrained_model_name_or_path="./load/zero123/${ZERO123_PREFIX}.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.05 \
    system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9 \
    system.guidance.cond_elevation_deg=$ELEVATION
