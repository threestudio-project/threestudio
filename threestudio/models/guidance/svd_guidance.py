import importlib
import os
from dataclasses import dataclass

import cv2
import imageio
import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, get_CPU_mem, get_GPU_mem
from threestudio.utils.typing import *
from tqdm import tqdm


def get_resizing_factor(
    desired_shape: Tuple[int, int], current_shape: Tuple[int, int]
) -> float:
    r_bound = desired_shape[1] / desired_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if aspect_r >= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r < 1.0:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)
    else:
        if aspect_r <= r_bound:
            factor = min(desired_shape) / min(current_shape)
        else:
            if aspect_r > 1:
                factor = max(desired_shape) / min(current_shape)
            else:
                factor = max(desired_shape) / max(current_shape)

    return factor


def resize_like(x, ref):
    H, W = ref.shape[-2], ref.shape[-1]
    h, w = x.shape[-2], x.shape[-1]
    rfs = get_resizing_factor((H, W), (h, w))
    resize_size = [int(np.ceil(rfs * s)) for s in (h, w)]
    top = (resize_size[0] - H) // 2
    left = (resize_size[1] - W) // 2
    x = torch.nn.functional.interpolate(x, resize_size, mode="area", antialias=False)
    x = TT.functional.crop(x, top=top, left=left, height=H, width=W)
    return x


@threestudio.register("svd-guidance")
class StableVideoDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        stable_research_path: str = "/admin/home-vikram/ROBIN/stable-research"
        pretrained_model_name_or_path: str = "prediction_stable_jucifer_3D_OBJ"
        cond_aug: float = 0.00
        num_steps: int = None  # 50
        height: int = 576
        guidance_scale: float = None  # 4.0

        vram_O: bool = True

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        noises_per_item: Optional[Any] = None

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        guidance_eval_freq: int = 0
        guidance_eval_dir: str = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading SVD ...")

        import sys

        sys.path.append(os.path.realpath(self.cfg.stable_research_path))
        from scripts.demo.video_denoiser import JucifierDenoiser

        self.model = JucifierDenoiser(
            self.cfg.pretrained_model_name_or_path,
            self.cfg.cond_aug,
            self.cfg.num_steps,
            self.cfg.guidance_scale,
            self.cfg.height,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.grad_clip_val: Optional[float] = None

        self.prepare_embeddings(self.cfg.cond_image_path)

        self.T = self.model.T
        self.num_steps = self.model.num_steps
        self.set_min_max_steps()  # set to default values

        self.count = 0

        threestudio.info(f"Loaded SVD!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_steps * min_step_percent)
        self.max_step = int(self.num_steps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(
        self, image_path: str, background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> None:
        # load cond image for Jucifier
        assert os.path.exists(image_path)
        self.model.process_cond_img(imageio.imread(image_path))

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        return self.model.encode_img(imgs)

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        **kwargs,
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.encode_images(rgb_BCHW)

        latents = torch.repeat_interleave(latents, self.noises_per_item, 0)
        batch_size = latents.shape[0]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size // self.T],
            dtype=torch.long,
            device=self.device,
        )

        guidance_eval = (
            self.cfg.guidance_eval_freq > 0
            and self.count % self.cfg.guidance_eval_freq == 0
        )

        with torch.no_grad():
            rgb_pred = self.model(latents, t, guidance_eval=guidance_eval)

        if guidance_eval:
            rgb_pred, rgb_i, rgb_d, rgb_eval = rgb_pred
            imageio.mimsave(
                os.path.join(
                    self.cfg.guidance_eval_dir,
                    f"guidance_eval_input_{self.count:05d}.mp4",
                ),
                rgb_i,
            )
            imageio.mimsave(
                os.path.join(
                    self.cfg.guidance_eval_dir,
                    f"guidance_eval_denoise_{self.count:05d}.mp4",
                ),
                rgb_d,
            )
            imageio.mimsave(
                os.path.join(
                    self.cfg.guidance_eval_dir,
                    f"guidance_eval_final_{self.count:05d}.mp4",
                ),
                rgb_eval,
            )
        self.count += 1

        # TODO CFG
        # TODO min_step, max_step
        # TODO grad w
        # # perform guidance
        # noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
        #     noise_pred_cond - noise_pred_uncond
        # )

        rgb_pred = resize_like(rgb_pred, rgb_BCHW)

        # w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = rgb_pred - rgb_BCHW
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # # loss = SpecifyGradient.apply(latents, grad)
        # # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        # target = (latents - grad).detach()
        # # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_sds = 0.5 * ((grad) ** 2).sum() / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            # "min_step": self.min_step,
            # "max_step": self.max_step,
            "cpu_mem": get_CPU_mem(),
            "gpu_mem": get_GPU_mem()[0],
        }

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

        if self.cfg.noises_per_item is not None:
            self.noises_per_item = np.floor(
                C(self.cfg.noises_per_item, epoch, global_step)
            ).astype(int)
        else:
            self.noises_per_item = 1

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def generate(
        self,
        image,  # image tensor [1, 3, H, W] in [0, 1]
        elevation=0,
        azimuth=0,
        camera_distances=0,  # new view params
        c_crossattn=None,
        c_concat=None,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        if c_crossattn is None:
            c_crossattn, c_concat = self.get_img_embeds(image)

        cond = self.get_cond(
            elevation, azimuth, camera_distances, c_crossattn, c_concat
        )

        imgs = self.gen_from_cond(cond, scale, ddim_steps, post_process, ddim_eta)

        return imgs

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def gen_from_cond(
        self,
        cond,
        scale=3,
        ddim_steps=50,
        post_process=True,
        ddim_eta=1,
    ):
        # produce latents loop
        B = cond["c_crossattn"][0].shape[0] // 2
        latents = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)

        for t in self.scheduler.timesteps:
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)[
                "prev_sample"
            ]

        imgs = self.decode_latents(latents)
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs
