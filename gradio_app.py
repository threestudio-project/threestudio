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

    def tolist(self):
        return [self.pid, self.progress, self.log, self.output_image, self.output_video]


EXP_ROOT_DIR = "outputs-gradio"
DEFAULT_PROMPT = "a delicious hamburger"
model_config = [
    ("DreamFusion (DeepFloyd-IF)", "configs/dreamfusion-if.yaml"),
    ("DreamFusion (Stable Diffusion)", "configs/dreamfusion-sd.yaml"),
    ("Fantasia3D (Stable Diffusion, Geometry Only)", "configs/fantasia3d.yaml"),
]
model_choices = [m[0] for m in model_config]
model_name_to_config = {m[0]: m[1] for m in model_config}


def load_model_config(model_name):
    return open(model_name_to_config[model_name]).read()


def on_model_selector_change(model_name):
    return load_model_config(model_name)


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

    log_path = os.path.join(trial_dir, "log")
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

    return status


def run(model_name: str, config: str, prompt: str, seed: int):
    # update status every 1 second
    status_update_interval = 1

    # save the config to a temporary file
    config_file = tempfile.NamedTemporaryFile(mode="w")

    with open(config_file.name, "w") as f:
        f.write(config)

    # manually assign the output directory, name and tag so that we know the trial directory
    name = os.path.basename(model_name_to_config[model_name]).split(".")[0]
    tag = datetime.now().strftime("@%Y%m%d-%H%M%S")
    trial_dir = os.path.join(EXP_ROOT_DIR, name, tag)
    alive_path = os.path.join(trial_dir, "alive")

    # spawn the training process
    process = subprocess.Popen(
        f"python launch.py --config {config_file.name} --train --gpu 0 --gradio trainer.enable_progress_bar=false".split()
        + [
            f'system.prompt_processor.prompt="{prompt}"',
            f'name="{name}"',
            f'tag="{tag}"',
            f"exp_root_dir={EXP_ROOT_DIR}",
            "use_timestamp=false",
            "trainer.max_steps=5000",
            "trainer.val_check_interval=100",
        ]
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
    try:
        print(f"Trying to kill process {pid} ...")
        os.kill(pid, signal.SIGKILL)
    except:
        pass
    # button status: Stop -> Reset
    return [
        gr.update(value="Reset", variant="secondary", visible=True),
        gr.update(visible=False),
    ]


def launch():
    with gr.Blocks() as demo:
        with gr.Row():
            pid = gr.State()
            with gr.Column(scale=1):
                # model selection dropdown
                model_selector = gr.Dropdown(
                    value=model_choices[0],
                    choices=model_choices,
                    label="Select a model",
                )

                # full config viewer
                with gr.Accordion("See full configurations", open=False):
                    config_editor = gr.Code(
                        value=load_model_config(model_selector.value),
                        language="yaml",
                        interactive=False,
                    )

                # prompt input
                prompt_input = gr.Textbox(value=DEFAULT_PROMPT, label="Input prompt")

                # seed slider
                seed_input = gr.Slider(
                    minimum=0, maximum=2147483647, value=0, step=1, label="seed"
                )

                # load config on model selection change
                model_selector.change(
                    fn=on_model_selector_change,
                    inputs=model_selector,
                    outputs=config_editor,
                )

                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)

            with gr.Column(scale=1):
                # generation status
                status = gr.Textbox(label="Status", lines=1, max_lines=1)

                # logs
                logs = gr.Textbox(label="Log", lines=10)

                # validation image display
                output_image = gr.Image()

                # testing video display
                output_video = gr.Video()

        run_event = run_btn.click(
            fn=run,
            inputs=[model_selector, config_editor, prompt_input, seed_input],
            outputs=[pid, status, logs, output_image, output_video, run_btn, stop_btn],
        )
        stop_btn.click(
            fn=stop_run, inputs=[pid], outputs=[run_btn, stop_btn], cancels=[run_event]
        )

    demo.queue().launch(server_name="0.0.0.0", server_port=12344)


def watch(
    pid: int, trial_dir: str, alive_timeout: int, wait_timeout: int, check_interval: int
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
                print(f"Process {pid} not exists, watcher exit.")
                exit(0)
            try:
                alive_timestamp = float(open(alive_path).read())
                if time.time() - alive_timestamp > alive_timeout:
                    os.kill(pid, signal.SIGKILL)
                    print(f"Alive timeout for process {pid}, killed.")
                    exit(0)
                else:
                    print(f"Process {pid} is alive.")
            except:
                print(f"Exception when checking alive for process {pid}, watcher exit.")
                exit(1)
            time.sleep(check_interval)

    # loop until alive file is found, or alive_timeout is reached
    loop_find_progress_file()
    # kill the process if it is not accessed for alive_timeout seconds
    loop_check_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", type=str, choices=["launch", "watch"])
    args, extra = parser.parse_known_args()
    if args.operation == "launch":
        launch()
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
