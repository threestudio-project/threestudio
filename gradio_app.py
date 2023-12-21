import argparse
import glob
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gradio as gr
import numpy as np
import psutil
import trimesh

from threestudio.utils.config import load_config
from threestudio.utils.typing import *


def tail(f, window=20):
    # Returns the last `window` lines of file `f`.
    if window == 0:
        return []

    BUFSIZ = 1024
    f.seek(0, 2)
    remaining_bytes = f.tell()
    size = window + 1
    block = -1
    data = []

    while size > 0 and remaining_bytes > 0:
        if remaining_bytes - BUFSIZ > 0:
            # Seek back one whole BUFSIZ
            f.seek(block * BUFSIZ, 2)
            # read BUFFER
            bunch = f.read(BUFSIZ)
        else:
            # file too small, start from beginning
            f.seek(0, 0)
            # only read what was not read
            bunch = f.read(remaining_bytes)

        bunch = bunch.decode("utf-8")
        data.insert(0, bunch)
        size -= bunch.count("\n")
        remaining_bytes -= BUFSIZ
        block -= 1

    return "\n".join("".join(data).splitlines()[-window:])


@dataclass
class ExperimentStatus:
    pid: Optional[int] = None
    progress: str = ""
    log: str = ""
    output_image: Optional[str] = None
    output_video: Optional[str] = None
    output_mesh: Optional[str] = None

    def tolist(self):
        return [
            self.pid,
            self.progress,
            self.log,
            self.output_image,
            self.output_video,
            self.output_mesh,
        ]


EXP_ROOT_DIR = "outputs-gradio"
DEFAULT_PROMPT = "a delicious hamburger"
model_name_config = [
    ("SJC (Stable Diffusion)", "configs/gradio/sjc.yaml"),
    ("DreamFusion (DeepFloyd-IF)", "configs/gradio/dreamfusion-if.yaml"),
    ("DreamFusion (Stable Diffusion)", "configs/gradio/dreamfusion-sd.yaml"),
    ("TextMesh (DeepFloyd-IF)", "configs/gradio/textmesh-if.yaml"),
    ("Latent-NeRF (Stable Diffusion)", "configs/gradio/latentnerf.yaml"),
    ("Fantasia3D (Stable Diffusion, Geometry Only)", "configs/gradio/fantasia3d.yaml"),
]
model_list = [m[0] for m in model_name_config]
model_config: Dict[str, Dict[str, Any]] = {}

for model_name, config_path in model_name_config:
    config = {"path": config_path}
    with open(config_path) as f:
        config["yaml"] = f.read()
    config["obj"] = load_config(
        config["yaml"],
        # set name and tag to dummy values to avoid creating new directories
        cli_args=[
            "name=dummy",
            "tag=dummy",
            "use_timestamp=false",
            f"exp_root_dir={EXP_ROOT_DIR}",
            "system.prompt_processor.prompt=placeholder",
        ],
        from_string=True,
    )
    model_config[model_name] = config


def on_model_selector_change(model_name):
    return [
        model_config[model_name]["yaml"],
        model_config[model_name]["obj"].system.guidance.guidance_scale,
    ]


def get_current_status(process, trial_dir, alive_path):
    status = ExperimentStatus()

    status.pid = process.pid

    # write the current timestamp to the alive file
    # the watcher will know the last active time of this process from this timestamp
    if os.path.exists(os.path.dirname(alive_path)):
        alive_fp = open(alive_path, "w")
        alive_fp.seek(0)
        alive_fp.write(str(time.time()))
        alive_fp.flush()

    log_path = os.path.join(trial_dir, "logs")
    progress_path = os.path.join(trial_dir, "progress")
    save_path = os.path.join(trial_dir, "save")

    # read current progress from the progress file
    # the progress file is created by GradioCallback
    if os.path.exists(progress_path):
        status.progress = open(progress_path).read()
    else:
        status.progress = "Setting up everything ..."

    # read the last 10 lines of the log file
    if os.path.exists(log_path):
        status.log = tail(open(log_path, "rb"), window=10)
    else:
        status.log = ""

    # get the validation image and testing video if they exist
    if os.path.exists(save_path):
        images = glob.glob(os.path.join(save_path, "*.png"))
        steps = [
            int(re.match(r"it(\d+)-0\.png", os.path.basename(f)).group(1))
            for f in images
        ]
        images = sorted(list(zip(images, steps)), key=lambda x: x[1])
        if len(images) > 0:
            status.output_image = images[-1][0]

        videos = glob.glob(os.path.join(save_path, "*.mp4"))
        steps = [
            int(re.match(r"it(\d+)-test\.mp4", os.path.basename(f)).group(1))
            for f in videos
        ]
        videos = sorted(list(zip(videos, steps)), key=lambda x: x[1])
        if len(videos) > 0:
            status.output_video = videos[-1][0]

        export_dirs = glob.glob(os.path.join(save_path, "*export"))
        steps = [
            int(re.match(r"it(\d+)-export", os.path.basename(f)).group(1))
            for f in export_dirs
        ]
        export_dirs = sorted(list(zip(export_dirs, steps)), key=lambda x: x[1])
        if len(export_dirs) > 0:
            obj = glob.glob(os.path.join(export_dirs[-1][0], "*.obj"))
            if len(obj) > 0:
                # FIXME
                # seems the gr.Model3D cannot load our manually saved obj file
                # here we load the obj and save it to a temporary file using trimesh
                mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
                trimesh.load(obj[0]).export(mesh_path.name)
                status.output_mesh = mesh_path.name

    return status


def run(
    model_name: str,
    config: str,
    prompt: str,
    guidance_scale: float,
    seed: int,
    max_steps: int,
    save_ckpt: bool,
    save_root: str,
):
    # update status every 1 second
    status_update_interval = 1

    # save the config to a temporary file
    config_file = tempfile.NamedTemporaryFile()

    with open(config_file.name, "w") as f:
        f.write(config)

    # manually assign the output directory, name and tag so that we know the trial directory
    name = os.path.basename(model_config[model_name]["path"]).split(".")[0]
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_dir = os.path.join(save_root, EXP_ROOT_DIR, name, tag)
    alive_path = os.path.join(trial_dir, "alive")

    # spawn the training process
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    process = subprocess.Popen(
        f"python launch.py --config {config_file.name} --train --gpu {gpu} --gradio trainer.enable_progress_bar=false".split()
        + [
            f'name="{name}"',
            f'tag="{tag}"',
            f"exp_root_dir={os.path.join(save_root, EXP_ROOT_DIR)}",
            "use_timestamp=false",
            f'system.prompt_processor.prompt="{prompt}"',
            f"system.guidance.guidance_scale={guidance_scale}",
            f"seed={seed}",
            f"trainer.max_steps={max_steps}",
        ]
        + (
            ["checkpoint.every_n_train_steps=${trainer.max_steps}"] if save_ckpt else []
        ),
    )

    # spawn the watcher process
    watch_process = subprocess.Popen(
        "python gradio_app.py watch".split()
        + ["--pid", f"{process.pid}", "--trial-dir", f"{trial_dir}"]
    )

    # update status (progress, log, image, video) every status_update_interval senconds
    # button status: Run -> Stop
    while process.poll() is None:
        time.sleep(status_update_interval)
        yield get_current_status(process, trial_dir, alive_path).tolist() + [
            gr.update(visible=False),
            gr.update(value="Stop", variant="stop", visible=True),
        ]

    # wait for the processes to finish
    process.wait()
    watch_process.wait()

    # update status one last time
    # button status: Stop / Reset -> Run
    status = get_current_status(process, trial_dir, alive_path)
    status.progress = "Finished."
    yield status.tolist() + [
        gr.update(value="Run", variant="primary", visible=True),
        gr.update(visible=False),
    ]


def stop_run(pid):
    # kill the process
    print(f"Trying to kill process {pid} ...")
    try:
        os.kill(pid, signal.SIGKILL)
    except:
        print(f"Exception when killing process {pid}.")
    # button status: Stop -> Reset
    return [
        # gr.update(
        #     value="Reset (refresh the page if in queue)",
        #     variant="secondary",
        #     visible=True,
        # just ask the user to refresh the page
        # ),
        gr.update(
            value="Please Refresh the Page",
            variant="secondary",
            visible=True,
            interactive=False,
        ),
        gr.update(visible=False),
    ]


def launch(
    port,
    listen=False,
    hf_space=False,
    self_deploy=False,
    save_ckpt=False,
    save_root=".",
):
    self_deploy = self_deploy or "TS_SELF_DEPLOY" in os.environ

    css = """
    #config-accordion, #logs-accordion {color: black !important;}
    .dark #config-accordion, .dark #logs-accordion {color: white !important;}
    .stop {background: darkred !important;}
    """

    with gr.Blocks(
        title="threestudio - Web Demo",
        theme=gr.themes.Monochrome(),
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            if hf_space:
                header = """
                # threestudio Text-to-3D Web Demo

                <div>
                <a style="display: inline-block;" href="https://github.com/threestudio-project/threestudio"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
                <a style="display: inline-block;" href="https://huggingface.co/spaces/bennyguo/threestudio?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space%20to%20skip%20the%20queue-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>
                </div>

                ### Usage
                - Select a model from the dropdown menu. If you duplicate this space and would like to use models based on DeepFloyd-IF, you need to [accept the license](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) and set `HUGGING_FACE_HUB_TOKEN` in `Repository secrets` in your space setting. You may also set `TS_SELF_DEPLOY` to enable changing arbitrary configurations.
                - Input a text prompt and hit the `Run` button to start.
                - Video and mesh export (not supported for SJC and Latent-NeRF) are available after the training process is finished.
                - **IMPORTANT NOTE: Keep this tab active when running the model.**
                """
            else:
                header = """
                # threestudio Text-to-3D Web Demo

                ### Usage
                - Select a model from the dropdown menu.
                - Input a text prompt and hit the `Run` button to start.
                - Video and mesh export (not supported for SJC and Latent-NeRF) are available after the training process is finished.
                - **IMPORTANT NOTE: Keep this tab active when running the model.**
                """
            gr.Markdown(header)

        with gr.Row(equal_height=False):
            pid = gr.State()
            with gr.Column(scale=1):
                # generation status
                status = gr.Textbox(
                    value="Hit the Run button to start.",
                    label="Status",
                    lines=1,
                    max_lines=1,
                )

                # model selection dropdown
                model_selector = gr.Dropdown(
                    value=model_list[0],
                    choices=model_list,
                    label="Select a model",
                )

                # prompt input
                prompt_input = gr.Textbox(value=DEFAULT_PROMPT, label="Input prompt")

                # guidance scale slider
                guidance_scale_input = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=model_config[model_selector.value][
                        "obj"
                    ].system.guidance.guidance_scale,
                    step=0.5,
                    label="Guidance scale",
                )

                # seed slider
                seed_input = gr.Slider(
                    minimum=0, maximum=2147483647, value=0, step=1, label="Seed"
                )

                max_steps_input = gr.Slider(
                    minimum=1,
                    maximum=20000 if self_deploy else 5000,
                    value=10000 if self_deploy else 5000,
                    step=1,
                    label="Number of training steps",
                )

                save_ckpt_checkbox = gr.Checkbox(
                    value=save_ckpt,
                    label="Save Checkpoints",
                    visible=False,
                    interactive=False,
                )

                save_root_state = gr.State(value=save_root)

                # full config viewer
                with gr.Accordion(
                    "See full configurations", open=False, elem_id="config-accordion"
                ):
                    config_editor = gr.Code(
                        value=model_config[model_selector.value]["yaml"],
                        language="yaml",
                        lines=10,
                        interactive=self_deploy,  # disable editing if in HF space
                    )

                # load config on model selection change
                model_selector.change(
                    fn=on_model_selector_change,
                    inputs=model_selector,
                    outputs=[config_editor, guidance_scale_input],
                    queue=False,
                )

                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)

            with gr.Column(scale=1):
                with gr.Accordion(
                    "See terminal logs", open=False, elem_id="logs-accordion"
                ):
                    # logs
                    logs = gr.Textbox(label="Logs", lines=10)

                # validation image display
                output_image = gr.Image(value=None, label="Image")

                # testing video display
                output_video = gr.Video(value=None, label="Video")

                # export mesh display
                output_mesh = gr.Model3D(value=None, label="3D Mesh")

        run_event = run_btn.click(
            fn=run,
            inputs=[
                model_selector,
                config_editor,
                prompt_input,
                guidance_scale_input,
                seed_input,
                max_steps_input,
                save_ckpt_checkbox,
                save_root_state,
            ],
            outputs=[
                pid,
                status,
                logs,
                output_image,
                output_video,
                output_mesh,
                run_btn,
                stop_btn,
            ],
            concurrency_limit=1,
        )
        stop_btn.click(
            fn=stop_run,
            inputs=[pid],
            outputs=[run_btn, stop_btn],
            cancels=[run_event],
            queue=False,
        )

    launch_args = {"server_port": port}
    if listen:
        launch_args["server_name"] = "0.0.0.0"
    demo.queue().launch(**launch_args)


def watch(
    pid: int,
    trial_dir: str,
    alive_timeout: int,
    wait_timeout: int,
    check_interval: int,
) -> None:
    print(f"Spawn watcher for process {pid}")

    def timeout_handler(signum, frame):
        exit(1)

    alive_path = os.path.join(trial_dir, "alive")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(wait_timeout)

    def loop_find_progress_file():
        while True:
            if not os.path.exists(alive_path):
                time.sleep(check_interval)
            else:
                signal.alarm(0)
                return

    def loop_check_alive():
        while True:
            if not psutil.pid_exists(pid):
                print(f"Process {pid} not exists, watcher exits.")
                cleanup_and_exit()
            try:
                alive_timestamp = float(open(alive_path).read())
            except:
                continue
            if time.time() - alive_timestamp > alive_timeout:
                print(f"Alive timeout for process {pid}, killed.")
                try:
                    os.kill(pid, signal.SIGKILL)
                except:
                    print(f"Exception when killing process {pid}.")
                cleanup_and_exit()
            time.sleep(check_interval)

    def cleanup_and_exit():
        exit(0)

    # loop until alive file is found, or alive_timeout is reached
    loop_find_progress_file()
    # kill the process if it is not accessed for alive_timeout seconds
    loop_check_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", type=str, choices=["launch", "watch"])
    args, extra = parser.parse_known_args()
    if args.operation == "launch":
        parser.add_argument("--listen", action="store_true")
        parser.add_argument("--hf-space", action="store_true")
        parser.add_argument("--self-deploy", action="store_true")
        parser.add_argument("--save-ckpt", action="store_true")  # unused
        parser.add_argument("--save-root", type=str, default=".")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        launch(
            args.port,
            listen=args.listen,
            hf_space=args.hf_space,
            self_deploy=args.self_deploy,
            save_ckpt=args.save_ckpt,
            save_root=args.save_root,
        )
    if args.operation == "watch":
        parser.add_argument("--pid", type=int)
        parser.add_argument("--trial-dir", type=str)
        parser.add_argument("--alive-timeout", type=int, default=10)
        parser.add_argument("--wait-timeout", type=int, default=10)
        parser.add_argument("--check-interval", type=int, default=1)
        args = parser.parse_args()
        watch(
            args.pid,
            args.trial_dir,
            alive_timeout=args.alive_timeout,
            wait_timeout=args.wait_timeout,
            check_interval=args.check_interval,
        )
