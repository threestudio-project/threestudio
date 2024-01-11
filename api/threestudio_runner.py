import argparse
import os
import time
import sys
import logging
from launch import main
from api.utils import *


# Taken from the argparser in threestudio/launch.py
def setup_threestudio_args(commandline: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")
    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )
    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )
    args, extras = parser.parse_known_args(commandline.strip().split(" "))

    for i in range(len(extras)):
        if "system.prompt_processor.prompt" in extras[i]:
            extras[i] = extras[i].replace("|", " ")

    return (args, extras)


def run_phase(args, phase_name):
    try:
        main(*args)
    except Exception as e:
        logging.error(f"{phase_name} error: {e}")
        raise RuntimeError(f"Internal error")


def launch_threestudio(request: Request, output_path: str):
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        raise Exception(f"{image_path} does not exist")

    init_image = None
    with open(image_path, "rb") as f:
        init_image = f.read()

    job_folder = sys.argv[2]

    try:
        prompt = request.prompt
        elevation = request.elevation_angle
        coarse_steps = request.coarse_steps
        refine_steps = request.refine_steps
        azimuth = request.azimuth_angle
        phase1_config = request.phase1_config
        phase2_config = request.phase2_config

    except Exception as e:
        raise Exception(f"Invalid arguments: {e}")

    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        raise Exception("CUDA_VISIBLE_DEVICES is not assigned")

    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")

    input_image_path = output_path + "input.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_image_path, "wb") as f:
        f.write(init_image)

    # TODO remove intermediate validation/weight savings

    prompt = prompt.replace(" ", "|")

    ####################################################
    ## Phase 1
    ####################################################
    phase1_output_ckpt_path = output_path + "Phase1/ckpts/last.ckpt"
    if os.path.isfile(phase1_output_ckpt_path):
        os.remove(phase1_output_ckpt_path)

    common_args = f" data.default_elevation_deg={elevation} data.default_azimuth_deg={azimuth} system.loggers.wandb.enable=False system.loggers.wandb.project=dummy system.loggers.wandb.name=dummy use_timestamp=false name={job_folder} trainer.val_check_interval=1000000 "
    phase1_args = setup_threestudio_args(
        f'--config ./configs/{phase1_config} --train trainer.max_steps={coarse_steps} tag=Phase1 data.image_path={input_image_path} system.prompt_processor.prompt="{prompt}"'
        + common_args
    )
    run_phase(phase1_args, "Phase 1")

    ####################################################
    ## Phase 2
    ####################################################
    phase2_output_ckpt_path = output_path + "Phase2/ckpts/last.ckpt"
    if os.path.isfile(phase2_output_ckpt_path):
        os.remove(phase2_output_ckpt_path)

    phase2_output_cfg_path = output_path + "Phase2/configs/parsed.yaml"
    if os.path.isfile(phase2_output_cfg_path):
        os.remove(phase2_output_cfg_path)

    phase2_args = setup_threestudio_args(
        f'--config ./configs/{phase2_config} --train trainer.max_steps={refine_steps} tag=Phase2 data.image_path={input_image_path} system.geometry_convert_from={phase1_output_ckpt_path} system.prompt_processor.prompt="{prompt}"'
        + common_args
    )
    run_phase(phase2_args, "Phase 2")

    ####################################################
    ## Phase 3
    ####################################################

    model_name = f"model_{int(time.time() * 100)}"

    phase3_args = setup_threestudio_args(
        f'--config ./configs/{phase2_config} --export tag=Phase3 system.exporter_type=mesh-exporter data.image_path={input_image_path} system.geometry_convert_from={phase2_output_ckpt_path} system.prompt_processor.prompt="{prompt}"'
        + common_args
    )
    run_phase(phase3_args, "Phase 3")
    phase3_output_path = output_path + f"Phase3/save/it0-export/"
    return phase3_output_path
