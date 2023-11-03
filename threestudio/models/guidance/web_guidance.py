import copy
import io
import json
import os
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import websocket
from PIL import Image
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


@threestudio.register("web-guidance")
class ControlNetGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        json_path: str = ""
        server_address: str = "127.0.0.1:8188"
        client_id: str = str(uuid.uuid4())

        rgb_name: str = ""
        cond_rgb_name: str = ""
        server_path: str = ""
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        input_size: int = 1024
        k_sampler_steps: int = 16

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Web ...")
        self.set_min_max_steps()  # set to default value
        self.prompt_json = json.load(open(self.cfg.json_path, "r"))

        threestudio.info(f"Loaded Web!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step_percent = min_step_percent
        self.max_step_percent = max_step_percent

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.cfg.client_id}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            "http://{}/prompt".format(self.cfg.server_address), data=data
        )
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            "http://{}/view?{}".format(self.cfg.server_address, url_values)
        ) as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(
            "http://{}/history/{}".format(self.cfg.server_address, prompt_id)
        ) as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)["prompt_id"]
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # Execution is done
            else:
                continue  # previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history["outputs"]:
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    images_output = []
                    for image in node_output["images"]:
                        image_data = self.get_image(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def save_image_cache(self, rgb, cond_rgb):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if rgb.shape[-1] == 4:
            rgb_filename = f"image_{timestamp}.png"
        else:
            rgb_filename = f"image_{timestamp}.jpg"
        cond_rgb_filename = f"cond_image_{timestamp}.jpg"

        rgb = (rgb[0].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cond_rgb = (cond_rgb[0].detach().clamp(0, 1).cpu().numpy() * 255).astype(
            np.uint8
        )
        # rgb = Image.fromarray(rgb)
        cond_rgb = Image.fromarray(cond_rgb)

        rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size))
        rgb[:, :, :3] = rgb[:, :, :3][:, :, ::-1]
        cond_rgb = cond_rgb.resize((self.cfg.input_size, self.cfg.input_size))

        cv2.imwrite(os.path.join(self.cfg.server_path, rgb_filename), rgb)
        cond_rgb.save(os.path.join(self.cfg.server_path, cond_rgb_filename))
        return rgb_filename, cond_rgb_filename

    def get_prompt_json(self, t, rgb_path, cond_rgb_path):
        prompt_json = copy.deepcopy(self.prompt_json)
        for key in prompt_json:
            if prompt_json[key]["class_type"] == "KSampler":
                if prompt_json[key]["inputs"]["steps"] == self.cfg.k_sampler_steps:
                    new_config = prompt_json[key]
                    new_config["inputs"]["denoise"] = t
                    prompt_json[key] = new_config
            if prompt_json[key]["class_type"] == "LoadImage":
                new_config = prompt_json[key]
                if new_config["inputs"]["image"] == self.cfg.rgb_name:
                    new_config["inputs"]["image"] = rgb_path
                elif new_config["inputs"]["image"] == self.cfg.cond_rgb_name:
                    new_config["inputs"]["image"] = cond_rgb_path
                prompt_json[key] = new_config
        return prompt_json

    def get_saveimage_id(self):
        prompt_json = self.prompt_json
        for key in prompt_json:
            if prompt_json[key]["class_type"] == "SaveImage":
                return key

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            int(self.min_step_percent * 1000),
            int(self.max_step_percent * 1000) + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        )
        t = t.item() / 1000

        rgb_path, cond_rgb_path = self.save_image_cache(rgb, cond_rgb)
        prompt_json = self.get_prompt_json(t, rgb_path, cond_rgb_path)
        ws = websocket.WebSocket()
        ws.connect(
            "ws://{}/ws?clientId={}".format(self.cfg.server_address, self.cfg.client_id)
        )
        images = self.get_images(ws, prompt_json)

        output_id = self.get_saveimage_id()
        image_data = images[output_id]
        image = Image.open(io.BytesIO(image_data[0]))
        image = image.resize((W, H))
        image = np.array(image)
        image = torch.FloatTensor(image).to(rgb.device) / 255
        image = image.unsqueeze(0)

        os.remove(os.path.join(self.cfg.server_path, rgb_path))
        os.remove(os.path.join(self.cfg.server_path, cond_rgb_path))

        return {"edit_images": image}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
