import os
import time

files = [
    "/fsx/proj-mod3d/threestudio/load/images/chess1_rgba.png",
]

for file in files:
    name = os.path.basename(file).split("_rgba.png")[0]
    with open(
        "/admin/home-vikram/git/threestudio/threestudio/scripts/run_zero123_sbatch.sh",
        "w",
    ) as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=vikky_{name}\n")
        f.write("#SBATCH --account=mod3d\n")
        f.write("#SBATCH --partition=g40\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --time=0-00:07:00\n")
        f.write("conda activate three\n")
        f.write("cd /admin/home-vikram/git/threestudio/\n")
        f.write(f"NAME={name}\n")
        # Phase 1
        f.write(
            "python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/XL_20230604.ckpt use_timestamp=False name=${NAME}_DEMO tag=XL_Phase1 system.loggers.wandb.enable=false system.loggers.wandb.project='voletiv-zero123XL-demo-NEW2' system.loggers.wandb.name=${NAME}_XL_Phase1\n"
        )
        # # Phase 1.5
        # f.write(
        #     "python launch.py --config configs/zero123-geometry.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}_DEMO/XL_Phase1/ckpts/last.ckpt system.guidance.pretrained_model_name_or_path=./load/zero123/XL_20230604.ckpt use_timestamp=False name=${NAME}_DEMO tag=XL_Phase1p5 system.loggers.wandb.enable=false system.loggers.wandb.project='voletiv-zero123XL-demo-NEW2' system.loggers.wandb.name=${NAME}_XL_Phase1p5\n"
        # )
    os.system(
        "sbatch /admin/home-vikram/git/threestudio/threestudio/scripts/run_zero123_sbatch.sh"
    )
    time.sleep(1)
