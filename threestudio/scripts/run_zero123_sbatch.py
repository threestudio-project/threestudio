import os
import time

files = [
    ["~/git/threestudio/load/images/bf1_rgba.png", -5.0, 0.0],
    ["~/git/threestudio/load/images/bf2_rgba.png", 0.0, 10.0],
    ["~/git/threestudio/load/images/bf3_rgba.png", 0.0, 0.0],
    ["~/git/threestudio/load/images/cr_ellie_rgba.png", 5.0, 70.0],
    ["~/git/threestudio/load/images/ew_punk1_rgba.png", 5.0, -70.0],
    ["~/git/threestudio/load/images/ew_punk2_rgba.png", 5.0, -70.0],
    ["~/git/threestudio/load/images/ew_punk3_rgba.png", 5.0, -70.0],
    ["~/git/threestudio/load/images/ew_punk4_rgba.png", 5.0, 30.0],
    ["~/git/threestudio/load/images/parrot1_rgba.png", -10.0, -20.0],
    ["~/git/threestudio/load/images/parrot2_rgba.png", -5.0, -50.0],
    ["~/git/threestudio/load/images/yy3_rgba.png", 5.0, 0.0],
    ["~/git/threestudio/load/images/yy4_rgba.png", 5.0, 0.0],
]

files = [
    ["~/git/threestudio/load/images/anya_front_rgba.png", 5.0],
    ["~/git/threestudio/load/images/baby_phoenix_on_ice_rgba.png", 15.0],
    ["~/git/threestudio/load/images/beach_house_1_rgba.png", 50.0],
    ["~/git/threestudio/load/images/beach_house_2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bollywood_actress_rgba.png", 5.0],
    ["~/git/threestudio/load/images/boy1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/boy2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bull2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/bust1_rgba.png", -5.0],
    ["~/git/threestudio/load/images/cactus_rgba.png", 5.0],
    ["~/git/threestudio/load/images/castle1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/castle2_rgba.png", 20.0],
    ["~/git/threestudio/load/images/catstatue_rgba.png", 5.0],
    ["~/git/threestudio/load/images/chess4_rgba.png", 5.0],
    ["~/git/threestudio/load/images/church_ruins_rgba.png", -10.0],
    ["~/git/threestudio/load/images/corgi_rgba.png", 5.0],
    ["~/git/threestudio/load/images/crystalpiano_rgba.png", 15.0],
    ["~/git/threestudio/load/images/dad3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dancer1_rgba.png", 10.0],
    ["~/git/threestudio/load/images/detective2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dinoskeleton2_rgba.png", -5.0],
    ["~/git/threestudio/load/images/doctor1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dodo1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dog1_rgba.png", 10.0],
    ["~/git/threestudio/load/images/dog3_rgba.png", 5.0],
    ["~/git/threestudio/load/images/dragon2_rgba.png", 10.0],
    ["~/git/threestudio/load/images/elephant2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/fantasy1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/firekeeper_rgba.png", 5.0],
    ["~/git/threestudio/load/images/fox1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/girl2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/globe1_rgba.png", 0.0],
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
    ["~/git/threestudio/load/images/km8_rgba.png", 15.0],
    ["~/git/threestudio/load/images/labrador6_rgba.png", 5.0],
    ["~/git/threestudio/load/images/labrador8_rgba.png", 5.0],
    ["~/git/threestudio/load/images/lawyer1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/lightningtree_rgba.png", 5.0],
    ["~/git/threestudio/load/images/mouse1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/nendoroid_obama1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/pilot1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/retriever6_rgba.png", 5.0],
    ["~/git/threestudio/load/images/retriever7_rgba.png", 5.0],
    ["~/git/threestudio/load/images/robot_rgba.png", 15.0],
    ["~/git/threestudio/load/images/sadhu1_rgba.png", 10.0],
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
    ["~/git/threestudio/load/images/woman5_rgba.png", 10.0],
    ["~/git/threestudio/load/images/yoga1_rgba.png", 5.0],
    ["~/git/threestudio/load/images/yoga2_rgba.png", 5.0],
    ["~/git/threestudio/load/images/yoga3_rgba.png", 5.0],
]


files = [
    ["~/git/threestudio/load/images/castle1_rgba.png", 0.0],
    ["~/git/threestudio/load/images/castle2_rgba.png", 20.0],
]

CONFIG = sai_multinoise
NAME = anya_front
MODEL = comb1
PROJECT = CHECK
ELEV = 5.0
WANDB = False

config = "3sai"
model = "elevcond2"
tag = ""
project = "Zero123_Dmitry_comp"
wandb = "true"


config = "3sai"
model = "sai"
tag = ""
project = "AMB"
wandb = "true"

config = "3sai"
model = "comb1"
tag = ""
project = "AMB"
wandb = "true"

config = "3sai"
model = "sai2"
tag = ""
project = "AMB"
wandb = "true"

config = "3sai"
model = "elevcond2"
tag = ""
project = "AMB"
wandb = "true"

config = "sai_multinoise_amb"
model = "sai"
tag = ""
project = "AMB"
wandb = "true"

config = "sai_multinoise_amb"
model = "comb1"
tag = "mn134_amb"
project = "AMB"
wandb = "True"

config = "sai_multinoise_amb"
model = "elevcond2"
tag = ""
project = "AMB"
wandb = "true"

config = "sai_multinoise_amb"
model = "sai2"
tag = ""
project = "AMB"
wandb = "true"

config = "sai_multinoise_amb_fast"
model = "sai2"
tag = "mn14_AMB_fast"
project = "AMB"
wandb = "True"

config = "sai_multinoise_amb_fast"
model = "comb1"
tag = "mn14_AMB_fast"
project = "AMB"
wandb = "True"


# for fileelev in files:
for fileelevazim in files:
    # for model in ["xl", "sai", "elevcond", "diffcolors", "nobottom"]:
    # file, elev = fileelev
    file, elev, azim = fileelevazim
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
        f.write("#SBATCH --time=0-00:30:00\n")
        f.write("#SBATCH --time=0-00:30:00\n")
        f.write("source  ~/miniconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate three\n")
        f.write("cd ~/git/threestudio/\n")
        f.write(f"NAME={name}\n")
        f.write(f"CONFIG={config}\n")
        f.write(f"MODEL={model}\n")
        f.write(f"ELEV={elev}\n")
        f.write(f"AZIM={azim}\n")
        f.write(f"PROJECT={project}\n")
        f.write(f"TAG={tag}\n")
        f.write(f"WANDB={wandb}\n")
        # Phase 1
        f.write(
            "python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=${PROJECT}/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_${AZIM}azim_${TAG} data.default_elevation_deg=${ELEV} data.default_azimuth_deg=${AZIM} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev_${AZIM}azim_${TAG}\n"
        )
    os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    time.sleep(1)

#######################################################
# 0.18215

config = "sai"
model = "sai2"
tag = ""
project = "Zero123_Dmitry_comp"
wandb = "true"

for fileelev in files[1:]:
    # for model in ["xl", "sai", "elevcond", "diffcolors", "nobottom"]:
    file, elev = fileelev
    name = os.path.basename(file).split("_rgba.png")[0]
    with open(
        os.path.expanduser(
            "~/git/threestudio_0.18215/threestudio/threestudio/scripts/zero123_sbatch.sh"
        ),
        "w",
    ) as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=vikky_{name}\n")
        f.write("#SBATCH --account=mod3d\n")
        f.write("#SBATCH --partition=g40\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --time=0-00:30:00\n")
        f.write("conda activate three\n")
        f.write("cd ~/git/threestudio_0.18215/threestudio/\n")
        f.write(f"NAME={name}\n")
        f.write(f"CONFIG={config}\n")
        f.write(f"MODEL={model}\n")
        f.write(f"ELEV={elev}\n")
        f.write(f"PROJECT={project}\n")
        f.write(f"TAG={tag}\n")
        f.write(f"WANDB={wandb}\n")
        # Phase 1
        f.write(
            "python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path=./load/zero123/zero123-${MODEL}.ckpt use_timestamp=false name=${PROJECT}/${NAME} tag=Phase1_${CONFIG}config_${MODEL}model_${ELEV}elev_${TAG} data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config_${MODEL}model_${ELEV}elev_${TAG}\n"
        )
    os.system(
        "sbatch ~/git/threestudio_0.18215/threestudio/threestudio/scripts/zero123_sbatch.sh"
    )
    time.sleep(1)


#######################################################
# Phase2

files = [
    [
        "~/git/threestudio/load/images/anya_front_rgba.png",
        5.0,
        "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )",
    ],
]

# Phase 2
phase1 = "/admin/home-vikram/git/threestudio/outputs/MAGIC123Zero123/anya_front/Phase1_sai_multinoise_ambconfig_comb1model_5.0elev_Phase1"
config = "magic123refine"
model = "comb1"
tag = "Phase2"
project = "MAGIC123Zero123"
wandb = "True"

for fileelevprompt in files:
    # for model in ["xl", "sai", "elevcond", "diffcolors", "nobottom"]:
    file, elev, prompt = fileelevprompt
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
        f.write("#SBATCH --time=0-00:30:00\n")
        f.write("conda activate three\n")
        f.write("cd ~/git/threestudio/\n")
        f.write(f"NAME={name}\n")
        f.write(f"CONFIG={config}\n")
        f.write(f"ELEV={elev}\n")
        f.write(f"PROJECT={project}\n")
        f.write(f"TAG={tag}\n")
        f.write(f"WANDB={wandb}\n")
        # Phase 2
        f.write(f"PROMPT={prompt}\n")
        f.write(f"PHASE1={phase1}\n")
        f.write(
            'python launch.py --config configs/zero123_${CONFIG}.yaml --train data.image_path=./load/images/fsx/${NAME}_rgba.png system.prompt_processor.prompt="${PROMPT}" system.geometry_convert_from=${PHASE1}/ckpts/last.ckpt use_timestamp=false name=${PROJECT}/${NAME} tag=Phase2_${CONFIG}config_${ELEV}elev_${TAG} data.default_elevation_deg=${ELEV} system.loggers.wandb.enable=${WANDB} system.loggers.wandb.project=${PROJECT} system.loggers.wandb.name=${NAME}_${CONFIG}config_${ELEV}elev_${TAG}\n'
        )
    os.system("sbatch ~/git/threestudio/threestudio/scripts/zero123_sbatch.sh")
    time.sleep(1)
