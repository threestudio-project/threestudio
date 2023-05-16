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


# config = {
#     "max_iters": 1000,
#     "seed": 42,
#     "scheduler": "cosine",
#     "mode": "latent",
#     "prompt_processor_type": "stable-diffusion-prompt-processor",
#     "prompt_processor": {
#         "prompt": "a pineapple",
#         "view_dependent_prompting": False,
#     },
#     "guidance_type": "stable-diffusion-guidance",
#     "guidance": {
#         "half_precision_weights": False,
#         "guidance_scale": 100.0,
#         "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
#         "grad_clip": None,
#     },
#     "image": {
#         "width": 64,
#         "height": 64,
#     },
# }

# deepfloyd

config = {
    "max_iters": 1000,
    "seed": 42,
    "scheduler": "cosine",
    "mode": "rgb",  # deepfloyd does not support latent optimization
    "prompt_processor_type": "deep-floyd-prompt-processor",
    "prompt_processor": {
        "prompt": "a photo of lion",
        "view_dependent_prompting": True,
        "use_perp_neg": True,
        "spawn": False,
    },
    "guidance_type": "deep-floyd-guidance",
    "guidance": {
        "half_precision_weights": True,
        "guidance_scale": 7.0,
        "pretrained_model_name_or_path": "DeepFloyd/IF-I-XL-v1.0",
        "grad_clip": None,
        "use_perp_neg": True,
    },
    "image": {
        "width": 64,
        "height": 64,
    },
}

seed_everything(config["seed"])

guidance = threestudio.find(config["guidance_type"])(config["guidance"])
prompt_processor = threestudio.find(config["prompt_processor_type"])(
    config["prompt_processor"]
)

n_images = 6
azimuth = torch.linspace(0, 90, n_images).to(guidance.device)
elevation = torch.zeros_like(azimuth)
camera_distance = torch.zeros_like(azimuth)

processor_output = prompt_processor(elevation, azimuth, camera_distance)

w, h = config["image"]["width"], config["image"]["height"]
mode = config["mode"]
if mode == "rgb":
    target = nn.Parameter(
        torch.rand(1, h, w, 3, device=guidance.device).repeat(n_images, 1, 1, 1)
    )
    # target = nn.Parameter(torch.rand(n_images, h, w, 3, device=guidance.device))
else:
    target = nn.Parameter(torch.randn(n_images, h, w, 4, device=guidance.device))

optimizer = torch.optim.AdamW([target], lr=1e-1, weight_decay=0)
num_steps = config["max_iters"]
scheduler = (
    get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps * 1.5))
    if config["scheduler"] == "cosine"
    else None
)

# add time to out_dir
timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
out_dir = os.path.join(
    "outputs", "perpneg", f"{config['prompt_processor']['prompt']}{timestamp}"
)
os.makedirs(out_dir, exist_ok=True)

plt.axis("off")

for step in tqdm(range(num_steps + 1)):
    optimizer.zero_grad()
    loss = guidance(
        rgb=target, processor_output=processor_output, rgb_as_latents=(mode != "rgb")
    )
    loss["sds"].backward()

    grad = target.grad
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    guidance.update_step(epoch=0, global_step=step)

    if step % 5 == 0:
        if mode == "rgb":
            rgb = target
            # vis_grad = grad[..., :3]
            # vis_grad_norm = grad.norm(dim=-1)
        else:
            rgb = guidance.decode_latents(target.permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )
            # vis_grad = grad
            # vis_grad_norm = grad.norm(dim=-1)

        # vis_grad_norm = vis_grad_norm / vis_grad_norm.max()
        # vis_grad = vis_grad / vis_grad.max()
        img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()
        # img_grad = vis_grad.clamp(0, 1).detach().squeeze(0).cpu().numpy()
        # img_grad_norm = vis_grad_norm.clamp(0, 1).detach().squeeze(0).cpu().numpy()

        fig, ax = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        for col in range(n_images):
            ax[col].imshow(img_rgb[col])
            ax[col].axis("off")
        plt.savefig(os.path.join(out_dir, f"{step:05d}.png"))
