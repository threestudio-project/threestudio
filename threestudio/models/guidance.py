from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils import randn_tensor
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *
from threestudio.utils.misc import C


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
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


@threestudio.register('stable-diffusion-guidance')
class StableDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
        use_xformers: bool = False
        guidance_scale: float = 100.
        grad_clip: Optional[Any] = None # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = 'sds'

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
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weights_dtype
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None

        print(f"[INFO] loaded stable diffusion!")

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
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
            self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents) # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.unet(
                latent_model_input, torch.cat([t] * 2), encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == 'sds':
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == 'sjc':
            w = 1
        elif self.cfg.weighting_strategy == 'fantasia3d':
            w = (self.alphas[t]**0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.cfg.weighting_strategy}")

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # latents.backward(grad, retain_graph=True)

        return {
            'sds': loss,
            'grad_norm': grad.norm(),
        }
    
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents    

    def decode_latents(self, latents: Float[Tensor, "B 4 H W"]) -> Float[Tensor, "B 3 512 512"]:
        latents = F.interpolate(latents, (64, 64), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    def update_step(self, epoch: int, global_step: int):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)    


@threestudio.register('sjc-guidance')
class ScoreJacobianGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
        use_xformers: bool = False
        guidance_scale: float = 100.
        grad_clip: Optional[float] = None
        half_precision_weights: bool = True
        var_red: bool = True
        min_step_percent: float = 0.01
        max_step_percent: float = 0.97

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

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weights_dtype,
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear",
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.grad_clip_val = None

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)
        self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)
        print(f"[INFO] loaded stable diffusion!")

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False
    ):
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (64, 64), mode="bilinear", align_corners=False)
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.01, 0.97) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        sigma = self.us[t]

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents) # TODO: use torch generator
            y = latents

            zs = y + sigma * noise
            scaled_zs = zs / torch.sqrt(1 + sigma ** 2)

            # pred noise
            latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            Ds = zs - sigma * noise_pred

            if self.cfg.var_red:
                grad = -(Ds - y) / sigma
            else:
                grad = -(Ds - zs) / sigma

        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # latents.backward(grad, retain_graph=True)

        return {
            'sds': loss,
        }
    
    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents    

    def decode_latents(self, latents: Float[Tensor, "B 4 H W"]) -> Float[Tensor, "B 3 512 512"]:
        latents = F.interpolate(latents, (64, 64), mode="bilinear", align_corners=False)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def update_step(self, epoch: int, global_step: int):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
