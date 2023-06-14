debugpy-run launch.py -- --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    system.freq.ref_or_zero123="accumulate" freq.guidance_eval=13 system.freq.guidance_eval=13 trainer.max_steps=5000

python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    system.freq.ref_or_zero123="alternate" freq.guidance_eval=13 system.freq.guidance_eval=13

python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 trainer.max_steps=5000

python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13

# CURRENT EXPERIMENTS


# higher learning rate, fewer epochs
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 trainer.max_steps=5000 \
    system.loss.lambda_3d_normal_smooth=0.1 system.optimizer.args.lr=0.03 trainer.max_steps=5000

python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.1 system.optimizer.args.lr=0.03 trainer.max_steps=10000

# go all in:
python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 trainer.max_steps=5000 \
    system.loss.lambda_3d_normal_smooth=0.1 system.optimizer.args.lr=0.03 trainer.max_steps=5000 \
    system.renderer.num_samples_per_ray=384 data.random_camera.height=96 data.random_camera.width=96 \
    data.random_camera.batch_size=6

python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.1 system.optimizer.args.lr=0.03 trainer.max_steps=10000 \
    system.renderer.num_samples_per_ray=384 data.random_camera.height=96 data.random_camera.width=96 \
    data.random_camera.batch_size=6




# lambda_3d_normal_smooth=0.1
python launch.py --config configs/zero123.yaml --train --gpu 4 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 trainer.max_steps=5000 \
    system.loss.lambda_3d_normal_smooth=0.1

python launch.py --config configs/zero123.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.1

# lambda_3d_normal_smooth=0.2
python launch.py --config configs/zero123.yaml --train --gpu 6 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 trainer.max_steps=5000 \
    system.loss.lambda_3d_normal_smooth=0.2

python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-gradient_accum" \
    data.image_path=./load/images/anya_front_rgba.png  system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2


## AGAIN

# go all in w/ lower SDS:

# zero123xl
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt"

python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt"

# test variable noise range
python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt"

python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt"

# just increase bs=3 - BROKEN! renders big bars
python launch.py --config configs/zero123.yaml --train --gpu 4 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=3" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=3

# increase bs=4
python launch.py --config configs/zero123.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

# increase bs=4,
#     min_step_percent: [0, 0.5, 0.1, 2000]  # (start_iter, start_val, end_val, end_iter)
#     max_step_percent: [0, 0.9, 0.5, 2000]
python launch.py --config configs/zero123.yaml --train --gpu 6 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.5,0.1,2000];max_step_pct=[0,0.9,0.5,2000]" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

#     min_step_percent: [0, 0.4, 0.1, 2000]
#     max_step_percent: [0, 0.8, 0.3, 2000]
python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.4,0.1,2000];max_step_pct=[0,0.8,0.3,2000]" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

# FASTEST EARLY CONVERGENCE SO FAR
#     min_step_percent: [0, 0.4, 0.1, 2000]
#     max_step_percent: [0, 0.8, 0.3, 2000]
# lr=0.1
python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.4,0.1,2000];max_step_pct=[0,0.8,0.3,2000];lr=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

#     min_step_percent: [0, 0.4, 0.1, 2000]
#     max_step_percent: [0, 0.8, 0.3, 2000]
# lr=0.03
python launch.py --config configs/zero123.yaml --train --gpu 6 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.4,0.1,2000];max_step_pct=[0,0.8,0.3,2000];lr=0.03" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.03

# CONVERGES AFTER ~1000 iterations
# keep noise higher
#     min_step_percent: [0, 0.6, 0.2, 2000]
#     max_step_percent: [0, 0.9, 0.5, 2000]
# lr=0.1
python launch.py --config configs/zero123.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.2,2000];max_step_pct=[0,0.9,0.5,2000];lr=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=2000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

# LIKE PREVIOUS, BUT TRY 1000 iterations - SEEMS TO CONVERGE PRETTY WELL
# keep noise higher
#     min_step_percent: [0, 0.6, 0.4, 1000]
#     max_step_percent: [0, 0.9, 0.7, 1000]
# lr=0.1
python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.4,1000];max_step_pct=[0,0.9,0.7,1000];lr=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

# LIKE PREVIOUS, BUT TRY lr=0.2: CONCLUSION: more unstable
# keep noise higher
#     min_step_percent: [0, 0.6, 0.4, 1000]
#     max_step_percent: [0, 0.9, 0.7, 1000]
# lr=0.2
python launch.py --config configs/zero123.yaml --train --gpu 6 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.4,1000];max_step_pct=[0,0.9,0.7,1000];lr=0.2" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.2

# lr=0.05
python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.4,1000];max_step_pct=[0,0.9,0.7,1000];lr=0.05" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.05

# lr=0.1 actually using this noise schedule:
# min_step_percent: [0, 0.6, 0.4, 2000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.7, 2000]
python launch.py --config configs/zero123.yaml --train --gpu 6 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.4,1000];max_step_pct=[0,0.9,0.7,1000];lr=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

# lr=0.1 actually using this noise schedule:
# min_step_percent: [0, 0.6, 0.4, 2000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.7, 2000]
# 3d_normal_smooth=0.05 (l3dns)
python launch.py --config configs/zero123.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.6,0.4,1000];max_step_pct=[0,0.9,0.7,1000];lr=0.1;l3dns=0.05" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.05 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

# More aggressive noise
python launch.py --config configs/zero123.yaml --train --gpu 4 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.8,0.4,1000];max_step_pct=[0,0.98,0.98,1000];lr=0.1;l3dns=0.05" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.05 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

# Start aggressive, then medium noise.
# also l3dns=0.1
python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;bs=4;min_step_pct=[0,0.7,0.2,1000];max_step_pct=[0,0.9,0.9,1000];lr=0.1;l3dns=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.1 trainer.max_steps=1000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1


#-------------------

# start again from "zero123xl_altern_franken1;variable_noise_range" but increase batch size
# ...BUT FORGOT TO RESTORE DEFAULTS... so using:
#    min_step_percent: [0, 0.7, 0.2, 2000]  # (start_iter, start_val, end_val, end_iter)
#    max_step_percent: [0, 0.9, 0.9, 2000]
python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range;bs=4" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

# like previous, but higher lr
python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range;bs=4;lr=0.1" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 system.optimizer.args.lr=0.1

#
python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range;bs=4;" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=13 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

## MORE ATTEMPTS

python launch.py --config configs/zero123.yaml --train --gpu 0 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range;bs=4;min/max=0.02/0.98" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

python launch.py --config configs/zero123.yaml --train --gpu 1 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range;bs=4;min/max=0.02/0.98" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=4000 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4

python launch.py --config configs/zero123.yaml --train --gpu 2 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_altern_franken1;variable_noise_range;bs=4;min/max=0.02/0.98;lr=0.05;max_step=1999" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="alternate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05


# Seems OK so far...
python launch.py --config configs/zero123.yaml --train --gpu 3 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range;bs=4;min/max=0.02/0.98;lr=0.05;max_step=1999" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05

# HARDCODED: white bg ratio=0.0
python launch.py --config configs/zero123.yaml --train --gpu 4 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range;bs=4;min/max=0.02/0.98;lr=0.05;max_step=1999;wbgr=0.0" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05

# truly random colors
python launch.py --config configs/zero123.yaml --train --gpu 4 system.loggers.wandb.enable=true system.loggers.wandb.project="claforte-zero123xl" \
    system.loggers.wandb.name="zero123xl_accum_franken1;variable_noise_range;bs=4;min/max=0.02/0.98;lr=0.05;max_step=1999;wbgr=0.5;random_bg" \
    data.image_path=./load/images/anya_front_rgba.png system.freq.ref_or_zero123="accumulate" system.freq.guidance_eval=37 \
    system.loss.lambda_3d_normal_smooth=0.2 trainer.max_steps=1999 \
    system.prompt_processor.pretrained_model_name_or_path="./load/zero123/zero123xl_last.ckpt" \
    data.random_camera.batch_size=4 data.random_camera.batch_size=4 system.optimizer.args.lr=0.05
