from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        # gt_grad = gt_grad * grad_scale
        return gt_grad, None


@threestudio.register('stable-diffusion-guidance')
class StableDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
        use_xformers: bool = False
        guidance_scale: float = 100.
        grad_clip: Optional[float] = None
        half_precision_weights: bool = True

    cfg: Config

    def configure(self) -> None:
        print(f"[INFO] loading stable diffusion...")

        weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32
        # Create model
        self.vae = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weights_dtype).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="unet", torch_dtype=weights_dtype).to(self.device)

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.use_xformers and is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        print(f"[INFO] loaded stable diffusion!")

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents) # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        if self.cfg.grad_clip is not None:
            grad = grad.clamp(-self.cfg.grad_clip, self.cfg.grad_clip)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        latents.backward(grad, retain_graph=True)

        return {}
    
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents    
