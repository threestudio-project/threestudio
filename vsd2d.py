import json
import math
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output, display
from ipywidgets import IntSlider, Output, interact
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import threestudio


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, -1)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# deepfloyd

# config = {
#     "max_iters": 200,
#     "seed": 3407,
#     "scheduler": "cosine",
#     "mode": "rgb",  # deepfloyd does not support latent optimization
#     "prompt_processor_type": "deep-floyd-prompt-processor",
#     "prompt_processor": {
#         "prompt": "lib:chimpanzee_banana",
#         "view_dependent_prompting": True,
#         "use_perp_neg": True,
#         "spawn": False,
#         # side back interpolation
#         # back -> side
#         # -f_sb(r_inter)
#         "f_sb": (4, 0.5, -2.426),
#         # -f_sb(r_inter)
#         # front negative
#         "f_fsb": (4, 0.5, -0.852),
#         # front side interpolation
#         # side -> front
#         # -f_fs(r_inter)
#         # front negative
#         # "f_fs": (4, 0.5, -2.426),
#         "f_fs": (-4, -0.5, 6.59),
#         # side negative
#         # -f_sf(1-r_inter)
#         "f_sf": (4, 0.5, -2.426),
#         # f_fs(0) == f_fsb(1)
#     },
#     "guidance_type": "deep-floyd-guidance",
#     "guidance": {
#         "half_precision_weights": True,
#         "guidance_scale": 7.0,
#         "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
#         "grad_clip": None,
#         "use_perp_neg": True,
#     },
#     "image": {
#         "width": 64,
#         "height": 64,
#     },
# }

config = {
    "max_iters": 6000,
    "seed": 42,
    "scheduler": None,
    "mode": "latent",
    "prompt_processor_type": "stable-diffusion-prompt-processor",
    "prompt_processor": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
        "prompt": "an astronaut is riding a horse",
        "spawn": False,
    },
    "guidance_type": "stable-diffusion-vsd-guidance",
    "guidance": {
        "half_precision_weights": True,
        "view_dependent_prompting": False,
        "guidance_scale": 7.5,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
        "pretrained_model_name_or_path_lora": "stabilityai/stable-diffusion-2-1-base",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "anneal_start_step": 2000,  # do not anneal
        "camera_condition_type": "extrinsics",
        "train_lora_repeat": 4,
    },
    "image": {
        "width": 64,
        "height": 64,
    },
    "n_particle": 16,
    "batch_size": 2,
    "n_accumulation_steps": 4,
    "save_interval": 50,
    "clip": False,
    "tanh": False,
}

seed_everything(config["seed"])

guidance = threestudio.find(config["guidance_type"])(config["guidance"]).cuda()
guidance.camera_embedding = guidance.camera_embedding.cuda()
prompt_processor = threestudio.find(config["prompt_processor_type"])(
    config["prompt_processor"]
)

n_images = config["n_particle"]
batch_size = config["batch_size"]

w, h = config["image"]["width"], config["image"]["height"]
mode = config["mode"]
if mode == "rgb":
    target = nn.Parameter(torch.rand(n_images, h, w, 3, device=guidance.device))
else:
    target = nn.Parameter(2 * torch.rand(n_images, h, w, 4, device=guidance.device) - 1)

optimizer = torch.optim.AdamW(
    [
        {"params": [target], "lr": 3e-2},
        {"params": guidance.parameters(), "lr": 1e-4},
    ],
    lr=3e-2,
    weight_decay=0,
)
num_steps = config["max_iters"]
scheduler = None

# add time to out_dir
timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
out_dir = os.path.join(
    "outputs", "vsd2d", f"{config['prompt_processor']['prompt']}{timestamp}"
)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

plt.axis("off")

elevation = torch.zeros([batch_size], device=guidance.device)
azimuth = torch.zeros([batch_size], device=guidance.device)
distance = torch.zeros([batch_size], device=guidance.device)
prompt_utils = prompt_processor()
save_interval = config["save_interval"]

mvp_mtx = torch.zeros([batch_size, 4, 4], device=guidance.device)
n_accumulation_steps = config["n_accumulation_steps"]

for step in tqdm(range(num_steps * n_accumulation_steps + 1)):
    # random select batch_size images from target with replacement
    particles = target[torch.randint(0, n_images, [batch_size])]
    if mode == "latent" and config["tanh"]:
        particles = torch.tanh(particles)

    loss_dict = guidance(
        rgb=particles,
        prompt_utils=prompt_utils,
        mvp_mtx=mvp_mtx,
        elevation=elevation,
        azimuth=azimuth,
        camera_distances=distance,
        c2w=mvp_mtx.clone(),
        rgb_as_latents=(mode != "rgb"),
    )
    loss = (loss_dict["loss_vsd"] + loss_dict["loss_lora"]) / n_accumulation_steps
    loss.backward()

    if (step + 1) % n_accumulation_steps == 0:
        actual_step = (step + 1) // n_accumulation_steps
        guidance.update_step(epoch=0, global_step=actual_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if mode == "latent" and config["clip"]:
            with torch.no_grad():
                particles.data = particles.data.clip(-1, 1)

        if actual_step % save_interval == 0:
            if mode == "rgb":
                rgb = target
            else:
                del loss
                torch.cuda.empty_cache()
                with torch.no_grad():
                    rgb = guidance.decode_latents(target.permute(0, 3, 1, 2)).permute(
                        0, 2, 3, 1
                    )

            img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()

            fig, ax = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
            for col in range(n_images):
                ax[col].imshow(img_rgb[col])
                ax[col].axis("off")
            plt.savefig(os.path.join(out_dir, f"{actual_step:05d}.png"))
