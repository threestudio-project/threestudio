import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from extern.zero123 import Zero123Pipeline
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("zero123-guidance")
class Zero123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "bennyguo/zero123-diffusers"
        enable_sequential_cpu_offload: bool = False
        guidance_scale: float = 5.0
        grad_clip: Optional[Any] = None
        half_precision_weights: bool = True

        height: int = 256
        width: int = 256

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero123 ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # need to make sure the pipeline file is in path
        sys.path.append("extern/")
        self.pipe = Zero123Pipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        for p in self.vae.parameters():
            p.requires_grad_(False)

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        self.clip_image_embeddings, self.image_latents = self.prepare_image_embeddings()

        del self.pipe.image_encoder
        cleanup()

        threestudio.info(f"Loaded Zero123!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_image_embeddings(
        self,
    ) -> Tuple[Float[Tensor, "1 1 768"], Float[Tensor, "1 4 Hl Wl"]]:
        assert os.path.exists(self.cfg.cond_image_path)
        rgba = torch.from_numpy(
            cv2.cvtColor(
                cv2.imread(self.cfg.cond_image_path, cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGRA2RGBA,
            ).astype(np.float32)
            / 255.0
        )
        assert rgba.shape[0] == rgba.shape[1], "Conditional image must be square."
        assert rgba.shape[2] == 4, "Conditional image must have 4 channels."

        # white background
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])

        rgb_pil = TF.to_pil_image(rgb.permute(2, 0, 1))
        processed_image = self.pipe.feature_extractor(
            images=rgb_pil, return_tensors="pt"
        ).pixel_values.to(device=self.device, dtype=self.weights_dtype)

        clip_image_embeddings = self.pipe.image_encoder(processed_image).image_embeds

        # FIXME: encoded latents should be multiplied with self.vae.config.scaling_factor
        # but zero123 was not trained this way
        rgb_pil.save("my.png")
        image_latents = self.vae.encode(
            TF.to_tensor(rgb_pil.resize((self.cfg.width, self.cfg.height))).to(
                device=self.device, dtype=self.weights_dtype
            )[None, ...]
            * 2.0
            - 1.0,
        ).latent_dist.mode()

        return clip_image_embeddings, image_latents

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 32,
        latent_width: int = 32,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(
                    rgb_BCHW,
                    (self.cfg.height // 8, self.cfg.width // 8),
                    mode="bilinear",
                    align_corners=False,
                )
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW,
                (self.cfg.height, self.cfg.width),
                mode="bilinear",
                align_corners=False,
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        camera_embeddings: Float[Tensor, "B 1 4"] = torch.stack(
            [
                torch.deg2rad(self.cfg.cond_elevation_deg - elevation),
                torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                camera_distances - self.cfg.cond_camera_distance,
            ],
            dim=-1,
        )[:, None, :]

        image_embeddings = self.pipe.clip_camera_projection(
            torch.cat(
                [
                    self.clip_image_embeddings.repeat(batch_size, 1, 1),
                    camera_embeddings,
                ],
                dim=-1,
            )
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat(
                [
                    torch.cat(
                        [latents_noisy, self.image_latents.repeat(batch_size, 1, 1, 1)],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            latents_noisy,
                            torch.zeros_like(self.image_latents).repeat(
                                batch_size, 1, 1, 1
                            ),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat(
                    [
                        image_embeddings.repeat(batch_size, 1, 1),
                        torch.zeros_like(image_embeddings).repeat(batch_size, 1, 1),
                    ],
                    dim=0,
                ),
            )

        # perform guidance
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_utils = {
                "t_orig": t,
                "latent_model_input": latent_model_input,
                "noise_pred": noise_pred,
                "image_embeddings": image_embeddings,
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(self, t_orig, latent_model_input, noise_pred, image_embeddings):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latent_model_input.shape[0] // 2)
            if self.cfg.max_items_eval > 0
            else latent_model_input.shape[0] // 2
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latent_model_input[:bs, :4]).permute(
            0, 2, 3, 1
        )

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latent_model_input[b : b + 1, :4], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                x_in = torch.cat(
                    [
                        torch.cat(
                            [latents, self.image_latents.repeat(1, 1, 1, 1)],
                            dim=1,
                        ),
                        torch.cat(
                            [
                                latents,
                                torch.zeros_like(self.image_latents).repeat(1, 1, 1, 1),
                            ],
                            dim=1,
                        ),
                    ],
                    dim=0,
                )
                t_in = torch.cat([t.reshape(1)] * 2).to(self.device)
                noise_pred = self.forward_unet(
                    x_in,
                    t_in,
                    torch.cat(
                        [
                            image_embeddings[b : b + 1],
                            torch.zeros_like(image_embeddings[b : b + 1]).repeat(
                                1, 1, 1
                            ),
                        ],
                        dim=0,
                    ),
                )
                # perform guidance
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

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
