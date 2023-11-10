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


class WebGuidance(BaseObject):
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
        print(t)

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


class WebDataset(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        dataroot: str = ""
        downsample_resolution: int = 1

    cfg: Config

    def configure(self) -> None:
        camera_dict = json.load(
            open(os.path.join(self.cfg.dataroot, "transforms.json"), "r")
        )
        scale = self.cfg.downsample_resolution
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        self.frames_moment = []
        self.frames_file_path = []
        self.frames_mask_path = []
        self.frames_intrinsic = []
        self.frames_bbox = []

        self.frames_t0 = []
        self.step = 0

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        threestudio.info("Loading frames...")
        self.n_frames = len(frames)

        for idx, frame in tqdm(enumerate(frames)):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"] / scale
            intrinsic[1, 1] = frame["fl_y"] / scale
            intrinsic[0, 2] = frame["cx"] / scale
            intrinsic[1, 2] = frame["cy"] / scale

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])

            if frame.__contains__("bbox"):
                self.frames_bbox.append(torch.FloatTensor(frame["bbox"]) / scale)

            self.frames_file_path.append(frame_path)
            if frame.__contains__("mask_path"):
                mask_path = os.path.join(self.cfg.dataroot, frame["mask_path"])
                self.frames_mask_path.append(mask_path)

            moment: Float[Tensor, "1"] = torch.zeros(1)
            if frame.__contains__("moment"):
                moment[0] = frame["moment"]
                if moment[0] < 1e-3:
                    self.frames_t0.append(idx)
            else:
                moment[0] = 0
                self.frames_t0.append(idx)

            self.frames_moment.append(moment)
        threestudio.info("Loaded frames.")

    def get_item(self, index=None):
        if index is None:
            if torch.randint(0, 1000, (1,)).item() % 2 == 0:
                index = torch.randint(0, self.n_frames, (1,)).item()
            else:
                t0_index = torch.randint(0, len(self.frames_t0), (1,)).item()
                index = self.frames_t0[t0_index]

        img = cv2.imread(self.frames_file_path[index])[:, :, ::-1]
        img = cv2.resize(img, (self.frame_w, self.frame_h))
        render_path = self.frames_file_path[index].replace("images", "images_render")
        if os.path.exists(render_path):
            print("get_render")
            render_img = cv2.imread(render_path)[:, :, ::-1]
            render_img = cv2.resize(render_img, (self.frame_w, self.frame_h))
        else:
            render_img = None
        if len(self.frames_mask_path) > 0:
            mask = cv2.imread(self.frames_mask_path[index])
            mask = cv2.resize(mask, (self.frame_w, self.frame_h))
        else:
            mask = np.ones_like(img)

        frame_img: Float[Tensor, "H W 3"] = torch.FloatTensor(img).unsqueeze(0) / 255
        if render_img is not None:
            frame_render: Float[Tensor, "H W 3"] = (
                torch.FloatTensor(render_img).unsqueeze(0) / 255
            )
        else:
            frame_render = None
        mask_img: Float[Tensor, "H W 3"] = torch.FloatTensor(mask).unsqueeze(0) / 255
        return_dict = {
            "index": index,
            "render_rgb": frame_render,
            "gt_rgb": frame_img,
            "frame_mask": mask_img,
            "height": self.frame_h,
            "width": self.frame_w,
            "moment": self.frames_moment[index],
            "file_path": self.frames_file_path[index],
        }
        if len(self.frames_bbox) > 0:
            return_dict.update(
                {
                    "frame_bbox": self.frames_bbox[index],
                }
            )
        return return_dict


def get_patch(batch, patch_size=512):
    origin_gt_rgb = batch["gt_rgb"]
    B, H, W, C = origin_gt_rgb.shape

    S = patch_size
    if batch.__contains__("frame_bbox"):
        bbox = batch["frame_bbox"]
        x1, y1, x2, y2 = (
            bbox[0].item(),
            bbox[1].item(),
            bbox[2].item(),
            bbox[3].item(),
        )
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        patch_x = int(min(max(0, center_x - S // 2), W - S - 1))
        patch_y = int(min(max(0, center_y - S // 2), H - S - 1))
    else:
        patch_x = W // 2 - S // 2
        patch_y = H // 2 - S // 2
    batch["patch_x"] = patch_x
    batch["patch_y"] = patch_y
    batch["patch_S"] = S
    return batch


def get_guidance_input(batch, refine_size=1024):
    origin_gt_rgb = batch["gt_rgb"]
    B, H, W, C = origin_gt_rgb.shape

    S = batch["patch_S"]
    patch_x = batch["patch_x"]
    patch_y = batch["patch_y"]
    if batch["render_rgb"] is None:
        batch["render_rgb"] = origin_gt_rgb[
            :, patch_y : patch_y + S, patch_x : patch_x + S
        ]
    batch["render_rgb"] = torch.nn.functional.interpolate(
        batch["render_rgb"].permute(0, 3, 1, 2),
        (refine_size, refine_size),
        mode="bilinear",
    ).permute(0, 2, 3, 1)

    if batch.__contains__("frame_mask"):
        guidance_input = torch.zeros(
            B, refine_size, refine_size, 4, device=origin_gt_rgb.device
        )
        mask = batch["frame_mask"][
            :, patch_y : patch_y + S, patch_x : patch_x + S
        ].clone()
        origin_patch_gt_rgb = origin_gt_rgb[
            :, patch_y : patch_y + S, patch_x : patch_x + S
        ]
        mask = torch.nn.functional.interpolate(
            mask.permute(0, 3, 1, 2),
            (refine_size, refine_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        origin_patch_gt_rgb = torch.nn.functional.interpolate(
            origin_patch_gt_rgb.permute(0, 3, 1, 2),
            (refine_size, refine_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

        guidance_input[:, :, :, :3] = batch["render_rgb"] * mask[
            :, :, :, :1
        ] + origin_patch_gt_rgb * (1 - mask[:, :, :, :1])
        guidance_input[:, :, :, 3] = 1.0 - mask[:, :, :, 0]
    else:
        guidance_input = batch["render_rgb"][
            :, patch_y : patch_y + S, patch_x : patch_x + S
        ]
    return guidance_input


if __name__ == "__main__":
    from threestudio.utils.config import load_config

    cfg = load_config(
        "/root/autodl-tmp/threestudio/configs/web_guidance/dynamic_gaussian_t4d_instruct_web.yaml"
    )
    guidance = WebGuidance(cfg.system.web_guidance)
    dataset = WebDataset(cfg.system.web_dataset)
    guidance.update_step(0, 0)

    os.makedirs(os.path.join(dataset.cfg.dataroot, "images_edit"), exist_ok=True)
    need_init = True
    if need_init:
        for i in tqdm(range(len(dataset.frames_file_path))):
            batch = dataset.get_item(i)
            if os.path.exists(batch["file_path"].replace("images", "images_edit")):
                continue
            batch = get_patch(batch)
            guidance_input = get_guidance_input(batch)
            result = guidance(
                guidance_input,
                batch["gt_rgb"][
                    :,
                    batch["patch_y"] : batch["patch_y"] + batch["patch_S"],
                    batch["patch_x"] : batch["patch_x"] + batch["patch_S"],
                ],
                None,
            )
            edit_image = result["edit_images"]
            edit_image.save(batch["file_path"].replace("images", "images_edit"))
    while True:
        batch = dataset.get_item()
        batch = get_patch(batch)
        guidance_input = get_guidance_input(batch)
        result = guidance(
            guidance_input,
            batch["gt_rgb"][
                :,
                batch["patch_y"] : batch["patch_y"] + batch["patch_S"],
                batch["patch_x"] : batch["patch_x"] + batch["patch_S"],
            ],
            None,
        )
        edit_image = result["edit_images"]
        edit_image.save(batch["file_path"].replace("images", "images_edit"))
