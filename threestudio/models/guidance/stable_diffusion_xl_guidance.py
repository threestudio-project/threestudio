from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import (
    StableDiffusionGuidance,
)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-xl-guidance")
class StableDiffusionXLGuidance(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-0.9"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        use_sjc: bool = False
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion XL ...")

        # assert self.cfg.half_precision_weights

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "tokenizer_2": None,
            "torch_dtype": self.weights_dtype,
            # "variant": "fp16",
            "use_safetensors": True,
        }
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        self.text_encoder_2_projection_dim = (
            self.pipe.text_encoder_2.config.projection_dim
        )

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.vae
        cleanup()

        # FIXME: float16 VAE produces NaNs
        # load float32 vae instead
        self.pipe.vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="vae",
        ).to(self.device)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            raise NotImplementedError

        if self.cfg.use_sjc:
            # score jacobian chaining use DDPM
            self.scheduler = DDPMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
                beta_start=0.00085,
                beta_end=0.0120,
                beta_schedule="scaled_linear",
            )
        else:
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
        if self.cfg.use_sjc:
            # score jacobian chaining need mu
            self.us: Float[Tensor, "..."] = torch.sqrt((1 - self.alphas) / self.alphas)

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion XL!")

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + self.text_encoder_2_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        cross_attention_kwargs: Optional[dict] = None,
        added_cond_kwargs: Optional[dict] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        ).sample.to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]
        H, W = latents.shape[-2] * 8, latents.shape[-1] * 8
        original_size = (H, W)
        target_size = (H, W)

        if prompt_utils.use_perp_neg:
            raise NotImplementedError
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            text_embeddings_pooled = prompt_utils.get_text_embeddings_pooled(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            add_text_embeds = text_embeddings_pooled
            add_time_ids = self._get_add_time_ids(
                original_size, (0, 0), target_size, dtype=latents.dtype
            ).to(latents)
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": torch.cat([add_time_ids] * 2, dim=0),
                }
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def compute_grad_sjc(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        batch_size = elevation.shape[0]
        H, W = latents.shape[-2] * 8, latents.shape[-1] * 8
        original_size = (H, W)
        target_size = (H, W)

        sigma = self.us[t]
        sigma = sigma.view(-1, 1, 1, 1)

        if prompt_utils.use_perp_neg:
            raise NotImplementedError
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            text_embeddings_pooled = prompt_utils.get_text_embeddings_pooled(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            add_text_embeds = text_embeddings_pooled
            add_time_ids = self._get_add_time_ids(
                original_size, (0, 0), target_size, dtype=latents.dtype
            ).to(latents)
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                y = latents

                zs = y + sigma * noise
                scaled_zs = zs / torch.sqrt(1 + sigma**2)

                # pred noise
                latent_model_input = torch.cat([scaled_zs] * 2, dim=0)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": torch.cat([add_time_ids] * 2, dim=0),
                }
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=added_cond_kwargs,
                )

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

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": scaled_zs,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 128 128"]
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (128, 128), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (1024, 1024), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sjc:
            grad, guidance_eval_utils = self.compute_grad_sjc(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )

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
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        raise NotImplementedError

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        raise NotImplementedError
