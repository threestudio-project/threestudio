import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from extern.ldm_zero123 import guidance
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-sdi-guidance")
class StableDiffusionSDIGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
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
        trainer_max_steps: int = 10000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/

        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4

        n_ddim_steps: int = 50

        # SDI parameters https://arxiv.org/abs/2405.15891
        enable_sdi: bool = True  # if false, sample noise randomly like in SDS
        inversion_guidance_scale: float = -7.5
        inversion_n_steps: int = 10
        inversion_eta: float = 0.3
        t_anneal: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

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
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.device
        )
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps, device=self.device)

        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.inverse_scheduler.set_timesteps(
            self.cfg.inversion_n_steps, device=self.device
        )
        self.inverse_scheduler.alphas_cumprod = (
            self.inverse_scheduler.alphas_cumprod.to(device=self.device)
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

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
        latent_height: int = 64,
        latent_width: int = 64,
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
    @torch.no_grad()
    def predict_noise(
        self,
        latents_noisy: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_scale: float = 1.0,
        text_embeddings: Optional[Float[Tensor, "..."]] = None,
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 4),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(-1, 1, 1, 1).to(
                    e_i_neg.device
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + guidance_scale * (e_pos + accum_grad)
        else:
            neg_guidance_weights = None

            if text_embeddings is None:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation,
                    azimuth,
                    camera_distances,
                    self.cfg.view_dependent_prompting,
                )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred, neg_guidance_weights, text_embeddings

    def ddim_inversion_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # 1. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = (
            self.inverse_scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.inverse_scheduler.initial_alpha_cumprod
        )
        alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.inverse_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.inverse_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.inverse_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.inverse_scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 3. Clip or threshold "predicted x_0"
        if self.inverse_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.inverse_scheduler.config.clip_sample_range,
                self.inverse_scheduler.config.clip_sample_range,
            )
        # 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        # 6. Add noise to the sample
        variance = self.scheduler._get_variance(prev_timestep, timestep) ** (0.5)
        prev_sample += self.cfg.inversion_eta * torch.randn_like(prev_sample) * variance

        return prev_sample

    def get_inversion_timesteps(self, invert_to_t, B):
        n_training_steps = self.inverse_scheduler.config.num_train_timesteps
        effective_n_inversion_steps = (
            self.cfg.inversion_n_steps
        )  # int((n_training_steps / invert_to_t) * self.cfg.inversion_n_steps)

        if self.inverse_scheduler.config.timestep_spacing == "leading":
            step_ratio = n_training_steps // effective_n_inversion_steps
            timesteps = (
                (np.arange(0, effective_n_inversion_steps) * step_ratio)
                .round()
                .copy()
                .astype(np.int64)
            )
            timesteps += self.inverse_scheduler.config.steps_offset
        elif self.inverse_scheduler.config.timestep_spacing == "trailing":
            step_ratio = n_training_steps / effective_n_inversion_steps
            timesteps = np.round(
                np.arange(n_training_steps, 0, -step_ratio)[::-1]
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )
        # use only timesteps before invert_to_t
        timesteps = timesteps[timesteps < int(invert_to_t)]

        # Roll timesteps array by one to reflect reversed origin and destination semantics for each step
        timesteps = np.concatenate([[int(timesteps[0] - step_ratio)], timesteps])
        timesteps = torch.from_numpy(timesteps).to(self.device)

        # Add the last step
        delta_t = int(
            random.random()
            * self.inverse_scheduler.config.num_train_timesteps
            // self.cfg.inversion_n_steps
        )
        last_t = torch.tensor(
            min(  # timesteps[-1] + self.inverse_scheduler.config.num_train_timesteps // self.inverse_scheduler.num_inference_steps,
                invert_to_t + delta_t,
                self.inverse_scheduler.config.num_train_timesteps - 1,
            ),
            device=self.device,
        )
        timesteps = torch.cat([timesteps, last_t.repeat([B])])
        return timesteps

    @torch.no_grad()
    def invert_noise(
        self,
        start_latents,
        invert_to_t,
        prompt_utils,
        elevation,
        azimuth,
        camera_distances,
    ):
        latents = start_latents.clone()
        B = start_latents.shape[0]

        timesteps = self.get_inversion_timesteps(invert_to_t, B)
        for t, next_t in zip(timesteps[:-1], timesteps[1:]):
            noise_pred, _, _ = self.predict_noise(
                latents,
                t.repeat([B]),
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                guidance_scale=self.cfg.inversion_guidance_scale,
            )
            latents = self.ddim_inversion_step(noise_pred, t, next_t, latents)

        # remap the noise from t+delta_t to t
        found_noise = self.get_noise_from_target(start_latents, latents, next_t)

        return latents, found_noise

    def get_noise_from_target(self, target, cur_xt, t):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise = (cur_xt - target * alpha_prod_t ** (0.5)) / (beta_prod_t ** (0.5))
        return noise

    def get_x0(self, original_samples, noise_pred, t):
        step_results = self.scheduler.step(
            noise_pred, t[0], original_samples, return_dict=True
        )
        if "pred_original_sample" in step_results:
            return step_results["pred_original_sample"]
        elif "denoised" in step_results:
            return step_results["denoised"]
        raise ValueError("Looks like the scheduler does not compute x0")

    @torch.no_grad()
    def compute_grad_sdi(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        call_with_defined_noise: Optional[Float[Tensor, "B 4 64 64"]] = None,
    ):
        if call_with_defined_noise is not None:
            noise = call_with_defined_noise.clone()
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
        elif self.cfg.enable_sdi:
            latents_noisy, noise = self.invert_noise(
                latents, t, prompt_utils, elevation, azimuth, camera_distances
            )
        else:
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

        noise_pred, neg_guidance_weights, text_embeddings = self.predict_noise(
            latents_noisy,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            guidance_scale=self.cfg.guidance_scale,
        )

        latents_denoised = self.get_x0(
            latents_noisy, noise_pred, t
        ).detach()  # (latents_noisy - sigma * noise_pred) / alpha

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances,
        }

        return latents_denoised, latents_noisy, noise, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        test_info=False,
        call_with_defined_noise=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = rgb
        else:
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

        target, noisy_img, noise, guidance_eval_utils = self.compute_grad_sdi(
            latents,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            call_with_defined_noise=call_with_defined_noise,
        )

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        # target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sdi = (
            0.5 * F.mse_loss(latents, target.detach(), reduction="mean") / batch_size
        )

        guidance_out = {
            "loss_sdi": loss_sdi,
            "grad_norm": (latents - target).norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if test_info:
            guidance_out["target"] = self.decode_latents(target)[0].permute(1, 2, 0)
            guidance_out["target_latent"] = target
            guidance_out["noisy_img"] = self.decode_latents(noisy_img)[0].permute(
                1, 2, 0
            )
            guidance_out["noise_img"] = self.decode_latents(noise)[0].permute(1, 2, 0)
            guidance_out["noise_pred"] = guidance_eval_utils["noise_pred"]
            return guidance_out

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(
                **guidance_eval_utils, prompt_utils=prompt_utils
            )
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
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        prompt_utils,
        elevation,
        azimuth,
        camera_distances,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
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
                noise_pred, _, _ = self.predict_noise(
                    latents,
                    t,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                    guidance_scale=1.0,
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

        if self.cfg.t_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            )  # progress percentage
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
