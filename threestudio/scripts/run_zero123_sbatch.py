import os
import time

files = [
    "~/git/threestudio/load/images/dog1_rgba.png",
    "~/git/threestudio/load/images/dragon2_rgba.png",
]

for file in files:
    name = os.path.basename(file).split("_rgba.png")[0]
    with open(
        os.path.expanduser("~/git/threestudio/threestudio/scripts/zero123_sbatch.sh"),
        "w",
    ) as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=vikky_{name}\n")
        f.write("#SBATCH --account=mod3d\n")
        f.write("#SBATCH --partition=g40\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --time=0-00:07:00\n")
        f.write("conda activate three\n")
        f.write("cd ~/git/threestudio/\n")
        f.write(f"NAME={name}\n")
        # Phase 1
        f.write(
            "python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png use_timestamp=true name=${NAME} tag=Phase1 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1\n"
        )
        # # Phase 1.5
        # f.write(
        #     "python launch.py --config configs/zero123-geometry.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5 system.loggers.wandb.enable=true system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1p5\n"
        # )
    os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    time.sleep(1)
