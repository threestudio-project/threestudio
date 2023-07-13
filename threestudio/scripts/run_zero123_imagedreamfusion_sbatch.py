import os
import time

files = [
    [
        "A simple 3D render of an alien",
        "/fsx/proj-mod3d/threestudio/load/images/alien1_rgba.png",
    ],
    [
        "A simple 3D render of a laughing boy",
        "/fsx/proj-mod3d/threestudio/load/images/boy2_rgba.png",
    ],
    [
        "A simple 3D render of a castle",
        "/fsx/proj-mod3d/threestudio/load/images/castle2_rgba.png",
    ],
    [
        "A simple 3D render of an Indian dad",
        "/fsx/proj-mod3d/threestudio/load/images/dad3_rgba.png",
    ],
    [
        "A simple 3D render of an Indian wedding dancer",
        "/fsx/proj-mod3d/threestudio/load/images/dancer1_rgba.png",
    ],
    [
        "A simple 3D render of a dinosaur skeleton",
        "/fsx/proj-mod3d/threestudio/load/images/dinoskeleton2_rgba.png",
    ],
    [
        "A simple 3D render of a stupid dodo",
        "/fsx/proj-mod3d/threestudio/load/images/dodo1_rgba.png",
    ],
    [
        "A simple 3D render of a friendly dog",
        "/fsx/proj-mod3d/threestudio/load/images/dog1_rgba.png",
    ],
    [
        "A simple 3D render of a dragon",
        "/fsx/proj-mod3d/threestudio/load/images/dragon2_rgba.png",
    ],
    [
        "A simple 3D render of a cute girl wearing wellington boots",
        "/fsx/proj-mod3d/threestudio/load/images/girl2_rgba.png",
    ],
    [
        "A simple 3D render of a globe",
        "/fsx/proj-mod3d/threestudio/load/images/globe1_rgba.png",
    ],
    [
        "A simple 3D render of groot plant",
        "/fsx/proj-mod3d/threestudio/load/images/grootplant_rgba.png",
    ],
    [
        "A simple 3D render of a hamburger",
        "/fsx/proj-mod3d/threestudio/load/images/hamburger_rgba.png",
    ],
    [
        "A simple 3D render of a horse",
        "/fsx/proj-mod3d/threestudio/load/images/horse_rgba.png",
    ],
    [
        "A simple 3D render of a nendoroid of obama",
        "/fsx/proj-mod3d/threestudio/load/images/nendoroid_obama1_rgba.png",
    ],
    [
        "A simple 3D render of a golden retriever",
        "/fsx/proj-mod3d/threestudio/load/images/retriever9_rgba.png",
    ],
    [
        "A simple 3D render of a robot",
        "/fsx/proj-mod3d/threestudio/load/images/robot_rgba.png",
    ],
    [
        "A simple 3D render of a black woman in police uniform",
        "/fsx/proj-mod3d/threestudio/load/images/w1_rgba.png",
    ],
    [
        "A simple 3D render of a woman in pilot uniform",
        "/fsx/proj-mod3d/threestudio/load/images/w5_rgba.png",
    ],
    [
        "A simple 3D render of an Indian monk",
        "/fsx/proj-mod3d/threestudio/load/images/sadhu1_rgba.png",
    ],
    [
        "A simple 3D render of a construction worker",
        "/fsx/proj-mod3d/threestudio/load/images/worker1_rgba.png",
    ],
]

for prompt, file in files:
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
        f.write(f"PROMPT='{prompt}'\n")
        # # Phase 1
        # f.write(
        #     "python launch.py --config configs/zero123.yaml --train data.image_path=./load/images/${NAME}_rgba.png use_timestamp=true name=${NAME} tag=Phase1 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1\n"
        # )
        # # Phase 1.5
        # f.write(
        #     "python launch.py --config configs/zero123-geometry.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5 system.loggers.wandb.enable=true system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1p5\n"
        # )
        # Phase 2
        f.write(
            "python launch.py --config configs/experimental/imagecondition_refine.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt=${PROMPT} system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=${NAME}_Phase2_refine # system.freq.guidance_eval=0 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase2\n"
        )
    os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    time.sleep(1)
