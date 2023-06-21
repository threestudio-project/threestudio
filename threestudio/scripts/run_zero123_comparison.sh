threestudio/scripts/run_zero123_phase.sh 0 anya_front XL_20230604 0
threestudio/scripts/run_zero123_phase.sh 0 anya_front 105000 0

threestudio/scripts/run_zero123_phase.sh 1 baby_phoenix_on_ice XL_20230604 20
threestudio/scripts/run_zero123_phase.sh 1 baby_phoenix_on_ice 105000 20

threestudio/scripts/run_zero123_phase.sh 2 beach_house_1 XL_20230604 50
threestudio/scripts/run_zero123_phase.sh 2 beach_house_1 105000 50

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 30
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 105000 30

threestudio/scripts/run_zero123_phase.sh 4 bollywood_actress XL_20230604 0
threestudio/scripts/run_zero123_phase.sh 4 bollywood_actress 105000 0


# First experiments used:
# system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.9

# These first 3 experiments used: `lambda_orient: [0, 1., 20., 5000]``

# More comparisons with beach_house_2
# 25X smaller lambda_3d_normal_smooth
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.loss.lambda_3d_normal_smooth=0.02

# 25X smaller lambda_3d_normal_smooth and lambda_depth
threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04

# 25X smaller lambda_depth
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.loss.depth=0.04

# The following experiments used: `lambda_orient: 1``


# 25X smaller lambda_3d_normal_smooth and lambda_depth, lambda_orient=0.1 (doesn't increase to 20)
threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

# increase noise ratio
threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.9 system.guidance.max_step_percent=0.9 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.98 system.guidance.max_step_percent=0.98 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.995 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.9 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.6 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.995 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.9 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.8 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.9 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.6 system.guidance.max_step_percent=0.85 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.4 system.guidance.max_step_percent=0.85 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.4 system.guidance.max_step_percent=0.95 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.85 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.6 system.guidance.max_step_percent=0.85 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.6 system.guidance.max_step_percent=0.7 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.3 system.guidance.max_step_percent=0.7 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.02 system.loss.depth=0.04 system.loss.lambda_orient=0.1

# 5X higher lambdas (i.e. 5X smaller than current defaults)
threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.998 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.1 system.loss.depth=0.2 system.loss.lambda_orient=0.5

# decrease lambda_sparsity 20X
threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.3 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.05 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.1 system.loss.depth=0.2 system.loss.lambda_orient=0.5

# disable sparsity, depth and orient supervision
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.2 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.0 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.1 system.loss.depth=0.0 system.loss.lambda_orient=0.0


threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.01 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.01 system.loss.depth=0.01 system.loss.lambda_orient=0.01

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.01 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.01 system.loss.depth=0.01 system.loss.lambda_orient=0.01

threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.1 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.01 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=1 system.loss.depth=0.01 system.loss.lambda_orient=0.01

threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.4 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.01 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.1 system.loss.depth=0.01 system.loss.lambda_orient=0.01

# decreased azimuth range, increased elevation range

threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.min_step_percent=0.4 system.guidance.max_step_percent=0.998 system.loss.lambda_sparsity=0.01 trainer.max_steps=19999 system.loss.lambda_3d_normal_smooth=0.1 system.loss.depth=0.01 system.loss.lambda_orient=0.01


# noise attenuation
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

# also changed the `azimuth_range: [-180, 180]` and `camera_distance_range: [3.4, 3.4]`
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=0.8 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=0.6 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=0.5 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=0.3 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 trainer.max_steps=1999

# more tests - vastly reduce lambda_depth
threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=0.5 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 system.loss.lambda_depth=0.01 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.8 system.guidance.max_step_percent=0.998 system.loss.lambda_depth=0.01 trainer.max_steps=1999

# CURRENT BASELINE
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.2 system.guidance.max_step_percent=0.95 system.loss.lambda_depth=0.01 trainer.max_steps=1999

# new Attempts
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.2 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 trainer.max_steps=1999

# lower sparsity 20X
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.2 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 trainer.max_steps=1999

threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 trainer.max_steps=1999

# NEW EXPERIMENTS:


# 5X longer
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 trainer.max_steps=9999

# 5X longer, lambda_3d_normal_smooth=5X smaller
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.1 trainer.max_steps=9999

# 5X longer, lambda_3d_normal_smooth=10X smaller
threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 trainer.max_steps=9999

# 5X longer, lambda_3d_normal_smooth=10X smaller, lambda_rgb=5X smaller
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.5 system.guidance.max_step_percent=0.85 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 trainer.max_steps=9999

# higher noise range
threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.9 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 trainer.max_steps=9999

# higher guidance scale (7)
threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.9 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 system.guidance.guidance_scale=7.0 trainer.max_steps=9999

# higher guidance scale (10)
threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 50 system.guidance.noise_attenuation=1.0 system.guidance.min_step_percent=0.7 system.guidance.max_step_percent=0.9 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 system.guidance.guidance_scale=10.0 trainer.max_steps=9999



# higher noise range
# min_step_percent: [0, 0.7, 0.05, 8000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.3, 8000]
threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 50 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 trainer.max_steps=9999

# lower elevation, and lower elevation range (elevation_range: [10, 80])
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=40.0 trainer.max_steps=9999


# lower elevation, and lower elevation range (elevation_range: [10, 80])
# less small lamba_rgb (2X smaller than previous default)
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 trainer.max_steps=9999

# start with wider noise range
# min_step_percent: [0, 0.5, 0.05, 8000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.3, 8000]
threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 trainer.max_steps=9999

# faster lr (lr=0.1) system.optimizer.args.lr=0.1
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# elevation = 40
threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 40 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# NEXT:

# elevation = 35
# DIVERGES TO WHITE!
threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 35 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# widen noise range:
# min_step_percent: [0, 0.3, 0.05, 8000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.8, 8000]
# DIVERGES TO WHITE!
threestudio/scripts/run_zero123_phase.sh 6 beach_house_2 XL_20230604 35 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# elevation = 30
# min_step_percent: [0, 0.5, 0.3, 10000]  # (start_iter, start_val, end_val, end_iter)
# max_step_percent: [0, 0.9, 0.8, 10000]
# DIVERGES TO WHITE!
threestudio/scripts/run_zero123_phase.sh 7 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# all previous lines included: system.optimizer.args.lr=0.05
# same as above, plus:
# lr: [0, 0.1, 0.01, 10000]
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999


# NEXT:

# all above included:
# default_camera_distance: 3.2
# camera_distance_range: [3.4, 3.4]
# default_fovy_deg: 20.0
# light_distance_range: [7.5, 10.0]
# widerfov
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999


# same as above except:
# default_camera_distance: 1.7
# camera_distance_range: [1.7, 1.7]
# widerfov2
threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 30 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# height: 256
# width: 256
# data.random_camera.height=192 data.random_camera.width=192
# data.random_camera.batch_size=3  (RENDERING BUG!)
# RUNS OUT OF VRAM!
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 30 data.height=256 data.random_camera.batch_size=3 data.width=256 data.random_camera.height=192 data.random_camera.width=192 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 30 data.height=256 data.random_camera.batch_size=2 data.width=256 data.random_camera.height=192 data.random_camera.width=192 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.1 trainer.max_steps=9999

# larger res = 256
# lower lr (system.optimizer.args.lr=0.03)
threestudio/scripts/run_zero123_phase.sh 4 beach_house_2 XL_20230604 30 data.height=256 data.random_camera.batch_size=2 data.width=256 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999

# 384x384 ref image
threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 30 data.height=384 data.random_camera.batch_size=2 data.width=384 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999

# same as above except restore default:
# min_step_percent: 0.02
# max_step_percent: 0.98
# 256_beach_house_2_rgba.png@20230621-013353
threestudio/scripts/run_zero123_phase.sh 0 beach_house_2 XL_20230604 30 data.height=384 data.random_camera.batch_size=2 data.width=384 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.01 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999

# try zero lambda_depth
# 256_beach_house_2_rgba.png@20230621-021356
threestudio/scripts/run_zero123_phase.sh 1 beach_house_2 XL_20230604 30 data.height=384 data.random_camera.batch_size=2 data.width=384 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.0 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999

# try lambda_depth=0.05
# 256_beach_house_2_rgba.png@20230621-021946
threestudio/scripts/run_zero123_phase.sh 2 beach_house_2 XL_20230604 30 data.height=384 data.random_camera.batch_size=2 data.width=384 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.05 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.05 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999

# reduce: system.loss.lambda_3d_normal_smooth=0.02
# 256_beach_house_2_rgba.png@20230621-022807
threestudio/scripts/run_zero123_phase.sh 3 beach_house_2 XL_20230604 30 data.height=384 data.random_camera.batch_size=2 data.width=384 data.random_camera.height=256 data.random_camera.width=256 system.loss.lambda_depth=0.05 system.loss.lambda_sparsity=0.1 system.loss.lambda_3d_normal_smooth=0.02 system.loss.lambda_rgb=100.0 system.optimizer.args.lr=0.03 trainer.max_steps=9999
