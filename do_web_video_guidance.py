import copy
import io
import json
import os
import shutil
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


def remove_directory(directory_path):
    try:
        # Use shutil.rmtree to remove the directory and its contents
        shutil.rmtree(directory_path)
        print(f"Successfully removed directory: {directory_path}")
    except Exception as e:
        print(f"Error while removing directory: {e}")


class WebGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        json_path: str = ""
        static_json_path: str = ""
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
        self.static_json = json.load(open(self.cfg.static_json_path, "r"))

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

    def save_image_cache(self, rgbs, cond_rgbs):
        B, H, W, C = rgbs.shape

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        rgb_dir_name = os.path.join(".threestudio_cache", f"images_{timestamp}")
        cond_rgb_dir_name = os.path.join(
            ".threestudio_cache", f"cond_images_{timestamp}"
        )
        os.makedirs(rgb_dir_name, exist_ok=True)
        os.makedirs(cond_rgb_dir_name, exist_ok=True)

        for i in range(B):
            if rgbs.shape[-1] == 4:
                rgb_filename = "image_%05d.png" % i
            else:
                rgb_filename = "image_%05d.jpg" % i
            cond_rgb_filename = "cond_image_%05d.jpg" % i

            rgb = (rgbs[i].detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            cond_rgb = (cond_rgbs[i].detach().clamp(0, 1).cpu().numpy() * 255).astype(
                np.uint8
            )
            # rgb = Image.fromarray(rgb)
            cond_rgb = Image.fromarray(cond_rgb)

            rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size))
            rgb[:, :, :3] = rgb[:, :, :3][:, :, ::-1]
            cond_rgb = cond_rgb.resize((self.cfg.input_size, self.cfg.input_size))

            cv2.imwrite(os.path.join(rgb_dir_name, rgb_filename), rgb)
            cond_rgb.save(os.path.join(cond_rgb_dir_name, cond_rgb_filename))
        return rgb_dir_name, cond_rgb_dir_name

    def get_prompt_json(self, prompt_json, t, rgb_path, cond_rgb_path):
        prompt_json = copy.deepcopy(prompt_json)
        for key in prompt_json:
            if prompt_json[key]["class_type"] == "KSampler":
                if prompt_json[key]["inputs"]["steps"] == self.cfg.k_sampler_steps:
                    new_config = prompt_json[key]
                    new_config["inputs"]["denoise"] = t
                    prompt_json[key] = new_config
            if prompt_json[key]["class_type"] == "LoadImagesFromDirectory":
                new_config = prompt_json[key]
                if new_config["inputs"]["directory"] == self.cfg.rgb_name:
                    new_config["inputs"]["directory"] = os.path.abspath(rgb_path)
                elif new_config["inputs"]["directory"] == self.cfg.cond_rgb_name:
                    new_config["inputs"]["directory"] = os.path.abspath(cond_rgb_path)
                prompt_json[key] = new_config
        return prompt_json

    def get_saveimage_id(self, prompt_json):
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

        if batch_size == 8:
            prompt_json = self.get_prompt_json(
                self.prompt_json, t, rgb_path, cond_rgb_path
            )
        else:
            prompt_json = self.get_prompt_json(
                self.static_json, t, rgb_path, cond_rgb_path
            )

        output_id = self.get_saveimage_id(prompt_json)

        while True:
            ws = websocket.WebSocket()
            ws.connect(
                "ws://{}/ws?clientId={}".format(
                    self.cfg.server_address, self.cfg.client_id
                )
            )
            images = self.get_images(ws, prompt_json)
            if images.__contains__(output_id):
                break
            else:
                print("Error!")

        image_data = images[output_id]
        image_list = []
        for i in range(batch_size):
            image = Image.open(io.BytesIO(image_data[i]))
            image = image.resize((W, H))
            image_list.append(image)

        remove_directory(rgb_path)
        remove_directory(cond_rgb_path)
        return {"edit_images": image_list}

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

        self.frames_cam_index = []
        self.frames_time_index = []
        self.cam_frames = {}

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

            self.frames_cam_index.append(int(frame["cam_index"]))
            self.frames_time_index.append(int(frame["time_index"]))
            if not self.cam_frames.__contains__(int(frame["cam_index"])):
                self.cam_frames[int(frame["cam_index"])] = {}
            self.cam_frames[int(frame["cam_index"])][int(frame["time_index"])] = idx

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
            mask_img: Float[Tensor, "H W 3"] = (
                torch.FloatTensor(mask).unsqueeze(0) / 255
            )

        frame_img: Float[Tensor, "H W 3"] = torch.FloatTensor(img).unsqueeze(0) / 255
        if render_img is not None:
            frame_render: Float[Tensor, "H W 3"] = (
                torch.FloatTensor(render_img).unsqueeze(0) / 255
            )
            print("frame_render!")
        else:
            frame_render = None
        return_dict = {
            "index": index,
            "render_rgb": frame_render,
            "gt_rgb": frame_img,
            "height": self.frame_h,
            "width": self.frame_w,
            "moment": self.frames_moment[index],
            "file_path": self.frames_file_path[index],
            "cam_index": self.frames_cam_index[index],
            "time_index": self.frames_time_index[index],
        }
        if len(self.frames_bbox) > 0:
            return_dict.update(
                {
                    "frame_bbox": self.frames_bbox[index],
                }
            )
        if len(self.frames_mask_path) > 0:
            return_dict["frame_mask"] = mask_img

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
        print("none render rgb")
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
        guidance_input = batch["render_rgb"]
    return guidance_input


if __name__ == "__main__":
    from threestudio.utils.config import load_config

    cfg = load_config(
        "/root/autodl-tmp/threestudio/configs/web_guidance/dynamic_web_video.yaml"
    )
    guidance = WebGuidance(cfg.system.web_guidance)
    dataset = WebDataset(cfg.system.web_dataset)
    guidance.update_step(0, 0)

    get_patch_size = 1024
    need_init = True
    default_video_batch = 8

    os.makedirs(os.path.join(dataset.cfg.dataroot, "images_edit"), exist_ok=True)
    if need_init:
        for cam_id in dataset.cam_frames:
            for frame_id in range(
                0, len(dataset.cam_frames[cam_id]), default_video_batch
            ):
                video_batch = min(
                    default_video_batch, len(dataset.cam_frames[cam_id]) - frame_id
                )
                guidance_input_list = []
                cond_input_list = []
                save_path_list = []
                valid_files = 0
                for b in range(video_batch):
                    batch_index = dataset.cam_frames[cam_id][frame_id + b]
                    batch = dataset.get_item(batch_index)
                    batch = get_patch(batch, get_patch_size)
                    guidance_input = get_guidance_input(batch)

                    cond_input = batch["gt_rgb"][
                        :,
                        batch["patch_y"] : batch["patch_y"] + batch["patch_S"],
                        batch["patch_x"] : batch["patch_x"] + batch["patch_S"],
                    ]
                    guidance_input_list.append(guidance_input)
                    cond_input_list.append(cond_input)
                    save_path_list.append(
                        batch["file_path"].replace("images", "images_edit")
                    )
                    if os.path.exists(
                        batch["file_path"].replace("images", "images_edit")
                    ):
                        valid_files += 1
                if valid_files == video_batch:
                    continue
                result = guidance(
                    torch.cat(guidance_input_list, dim=0),
                    torch.cat(cond_input_list, dim=0),
                    None,
                )
                edit_images = result["edit_images"]
                for b in range(video_batch):
                    edit_images[b].save(save_path_list[b])
    while True:
        if os.path.exists(os.path.join(dataset.cfg.dataroot, "status.json")):
            status_json = json.load(
                open(os.path.join(dataset.cfg.dataroot, "status.json"), "r")
            )
            global_step = status_json["global_step"]
        else:
            global_step = 0
        print(global_step)
        guidance.update_step(0, global_step)
        batch = dataset.get_item()
        cam_id = batch["cam_index"]
        t_id = batch["time_index"]
        for frame_id in range(
            t_id, len(dataset.cam_frames[cam_id]), default_video_batch
        ):
            video_batch = min(
                default_video_batch, len(dataset.cam_frames[cam_id]) - frame_id
            )
            guidance_input_list = []
            cond_input_list = []
            save_path_list = []
            for b in range(video_batch):
                batch_index = dataset.cam_frames[cam_id][frame_id + b]
                batch = dataset.get_item(batch_index)
                batch = get_patch(batch, get_patch_size)
                guidance_input = get_guidance_input(batch)
                cond_input = batch["gt_rgb"][
                    :,
                    batch["patch_y"] : batch["patch_y"] + batch["patch_S"],
                    batch["patch_x"] : batch["patch_x"] + batch["patch_S"],
                ]
                guidance_input_list.append(guidance_input)
                cond_input_list.append(cond_input)
                save_path_list.append(
                    batch["file_path"].replace("images", "images_edit")
                )
            result = guidance(
                torch.cat(guidance_input_list, dim=0),
                torch.cat(cond_input_list, dim=0),
                None,
            )
            edit_images = result["edit_images"]
            for b in range(video_batch):
                edit_images[b].save(save_path_list[b])
            break
