import os
import time

files = [
    ["~/git/threestudio/load/images/anya_front_rgba.png", 5.0],
    ["~/git/threestudio/load/images/baby_phoenix_on_ice_rgba.png", 5.0],
    ["~/git/threestudio/load/images/beach_house_1_rgba.png", 30.0],
    ["~/git/threestudio/load/images/beach_house_2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bollywood_actress_rgba.png", 5.0],
    ["~/git/threestudio/load/images/boy1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/boy2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bull2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bust1_rgba.png", -5.0],
    ["~/git/threestudio/load/images/cactus_rgba.png", 5.0],
    ["~/git/threestudio/load/images/castle1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/castle2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/catstatue_rgba.png", 5.0],
    ["~/git/threestudio/load/images/chess4_rgba.png", 5.0],
    ["~/git/threestudio/load/images/church_ruins_rgba.png", -5.0],
    ["~/git/threestudio/load/images/corgi_rgba.png", 5.0],
    ["~/git/threestudio/load/images/crystalpiano_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dad3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dancer1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/detective2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dinoskeleton2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/doctor1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dodo1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dog1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dog3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dragon2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/elephant2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/fantasy1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/firekeeper_rgba.png", 5.0],
    ["~/git/threestudio/load/images/fox1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/girl2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/globe1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/grootplant_rgba.png", 5.0],
    ["~/git/threestudio/load/images/hero2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/hiker1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/hiker2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/hiker3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/hiker4_rgba.png", 5.0],
    ["~/git/threestudio/load/images/horse2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/invention2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/km4_rgba.png", 5.0],
    ["~/git/threestudio/load/images/km7_rgba.png", 5.0],
    ["~/git/threestudio/load/images/km8_rgba.png", 5.0],
    ["~/git/threestudio/load/images/labrador6_rgba.png", 5.0],
    ["~/git/threestudio/load/images/labrador8_rgba.png", 5.0],
    ["~/git/threestudio/load/images/lawyer1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/lightningtree_rgba.png", 5.0],
    ["~/git/threestudio/load/images/mouse1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/nendoroid_obama1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/pilot1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/retriever6_rgba.png", 5.0],
    ["~/git/threestudio/load/images/retriever7_rgba.png", 5.0],
    ["~/git/threestudio/load/images/robot_rgba.png", 5.0],
    ["~/git/threestudio/load/images/sadhu1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/sadhu2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/sarasvatee1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/sarasvatee2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/sofa_rgba.png", 5.0],
    ["~/git/threestudio/load/images/teapot_rgba.png", 5.0],
    ["~/git/threestudio/load/images/teddy_rgba.png", 5.0],
    ["~/git/threestudio/load/images/temple1_rgba.png", 30.0],
    ["~/git/threestudio/load/images/temple2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/temple3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/thorhammer_rgba.png", 5.0],
    ["~/git/threestudio/load/images/woman1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/woman3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/woman5_rgba.png", 5.0],
    ["~/git/threestudio/load/images/yoga1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/yoga2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/yoga3_rgba.png", 5.0],
]

config = "sai"
model = "sai"

for fileelev in files[1:]:
    file, elev = fileelev
    config = "sai"
    model = "sai"
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
        f.write("#SBATCH --time=0-00:15:00\n")
        f.write("conda activate three\n")
        f.write("cd ~/git/threestudio/\n")
        f.write(f"NAME={name}\n")
        f.write(f"CONFIG={config}\n")
        f.write(f"MODEL={model}\n")
        f.write(f"ELEV={elev}\n")
        # Phase 1
        f.write(
            "python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=SAI/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=true system.loggers.wandb.project='zero123_SAIconfig_comp' system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev\n"
        )
    os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    time.sleep(1)
    # ###########################
    # config = "sai"
    # model = "xl"
    # name = os.path.basename(file).split("_rgba.png")[0]
    # with open(
    #     os.path.expanduser("~/git/threestudio/threestudio/scripts/zero123_sbatch.sh"),
    #     "w",
    # ) as f:
    #     f.write("#!/bin/bash\n")
    #     f.write(f"#SBATCH --job-name=vikky_{name}\n")
    #     f.write("#SBATCH --account=mod3d\n")
    #     f.write("#SBATCH --partition=g40\n")
    #     f.write("#SBATCH --gpus=1\n")
    #     f.write("#SBATCH --time=0-00:15:00\n")
    #     f.write("conda activate three\n")
    #     f.write("cd ~/git/threestudio/\n")
    #     f.write(f"NAME={name}\n")
    #     f.write(f"CONFIG={config}\n")
    #     f.write(f"MODEL={model}\n")
    #     f.write(f"ELEV={elev}\n")
    #     # Phase 1
    #     f.write(
    #         "python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=SAI/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=true system.loggers.wandb.project='zero123_SAIconfig_comp' system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev\n"
    #     )
    # os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    # time.sleep(1)
    # ###########################
    # config = "sai"
    # model = "orig"
    # name = os.path.basename(file).split("_rgba.png")[0]
    # with open(
    #     os.path.expanduser("~/git/threestudio/threestudio/scripts/zero123_sbatch.sh"),
    #     "w",
    # ) as f:
    #     f.write("#!/bin/bash\n")
    #     f.write(f"#SBATCH --job-name=vikky_{name}\n")
    #     f.write("#SBATCH --account=mod3d\n")
    #     f.write("#SBATCH --partition=g40\n")
    #     f.write("#SBATCH --gpus=1\n")
    #     f.write("#SBATCH --time=0-00:15:00\n")
    #     f.write("conda activate three\n")
    #     f.write("cd ~/git/threestudio/\n")
    #     f.write(f"NAME={name}\n")
    #     f.write(f"CONFIG={config}\n")
    #     f.write(f"MODEL={model}\n")
    #     f.write(f"ELEV={elev}\n")
    #     # Phase 1
    #     f.write(
    #         "python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=SAI/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=true system.loggers.wandb.project='zero123_SAIconfig_comp' system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev\n"
    #     )
    # os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    # time.sleep(1)
    # ###########################

    # /admin/home-vikram/git/threestudio/outputs/SAI/temple2/Phase1_SAI_new_elev5_EXP9_lr0.01_dmitryFix_new/configs/raw.yaml
    # new, elev 10
    # f.write(
    #     "python launch.py --config configs/zero123_sai.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png data.default_elevation_deg=-10.0 use_timestamp=false name=SAI/${NAME} tag=Phase1_SAI_new_elev5_EXP2_lr0.001 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1\n"
    # )
    # original, elev 5
    # f.write(
    #     "python launch.py --config configs/zero123_orig.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png data.default_elevation_deg=5.0 use_timestamp=false name=${NAME} tag=Phase1_orig_elev5 system.loggers.wandb.enable=false system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1\n"
    # )
    # # Phase 1.5
    # f.write(
    #     "python launch.py --config configs/zero123-geometry.yaml --train data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5 system.loggers.wandb.enable=true system.loggers.wandb.project='zero123' system.loggers.wandb.name=${NAME}_Phase1p5\n"
    # )
