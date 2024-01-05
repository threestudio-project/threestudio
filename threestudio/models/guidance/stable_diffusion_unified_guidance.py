import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.networks import ToDTypeWrapper
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, enable_gradient, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-unified-guidance")
class StableDiffusionUnifiedGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        # guidance type, in ["sds", "vsd"]
        guidance_type: str = "sds"

        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        guidance_scale: float = 100.0
        weighting_strategy: str = "dreamfusion"
        view_dependent_prompting: bool = True

        min_step_percent: Any = 0.02
        max_step_percent: Any = 0.98
        grad_clip: Optional[Any] = None

        return_rgb_1step_orig: bool = False
        return_rgb_multistep_orig: bool = False
        n_rgb_multistep_orig_steps: int = 4

        # TODO
        # controlnet
        controlnet_model_name_or_path: Optional[str] = None
        preprocessor: Optional[str] = None
        control_scale: float = 1.0

        # TODO
        # lora
        lora_model_name_or_path: Optional[str] = None

        # efficiency-related configurations
        half_precision_weights: bool = True
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        # VSD configurations, only used when guidance_type is "vsd"
        vsd_phi_model_name_or_path: Optional[str] = None
        vsd_guidance_scale_phi: float = 1.0
        vsd_use_lora: bool = True
        vsd_lora_cfg_training: bool = False
        vsd_lora_n_timestamp_samples: int = 1
        vsd_use_camera_condition: bool = True
        # camera condition type, in ["extrinsics", "mvp", "spherical"]
        vsd_camera_condition_type: Optional[str] = "extrinsics"

        # HiFA configurations: https://hifa-team.github.io/HiFA-site/
        sqrt_anneal: bool = (
            False  # requires setting min_step_percent=0.3 to work properly
        )
        trainer_max_steps: int = 25000
        use_img_loss: bool = True  # works with most cases

    cfg: Config

    def configure(self) -> None:
        self.min_step: Optional[int] = None
        self.max_step: Optional[int] = None
        self.grad_clip_val: Optional[float] = None

        @dataclass
        class NonTrainableModules:
            pipe: StableDiffusionPipeline
            pipe_phi: Optional[StableDiffusionPipeline] = None
            controlnet: Optional[ControlNetModel] = None

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        threestudio.info(f"Loading Stable Diffusion ...")

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        self.prepare_pipe(pipe)
        self.configure_pipe_token_merging(pipe)

        # phi network for VSD
        # introduce two trainable modules:
        # - self.camera_embedding
        # - self.lora_layers
        pipe_phi = None

        # if the phi network shares the same unet with the pretrain network
        # we need to pass additional cross attention kwargs to the unet
        self.vsd_share_model = (
            self.cfg.guidance_type == "vsd"
            and self.cfg.vsd_phi_model_name_or_path is None
        )
        if self.cfg.guidance_type == "vsd":
            if self.cfg.vsd_phi_model_name_or_path is None:
                pipe_phi = pipe
            else:
                pipe_phi = StableDiffusionPipeline.from_pretrained(
                    self.cfg.vsd_phi_model_name_or_path,
                    **pipe_kwargs,
                ).to(self.device)
                self.prepare_pipe(pipe_phi)
                self.configure_pipe_token_merging(pipe_phi)

            # set up camera embedding
            if self.cfg.vsd_use_camera_condition:
                if self.cfg.vsd_camera_condition_type in ["extrinsics", "mvp"]:
                    self.camera_embedding_dim = 16
                elif self.cfg.vsd_camera_condition_type == "spherical":
                    self.camera_embedding_dim = 4
                else:
                    raise ValueError("Invalid camera condition type!")

                # FIXME: hard-coded output dim
                self.camera_embedding = ToDTypeWrapper(
                    TimestepEmbedding(self.camera_embedding_dim, 1280),
                    self.weights_dtype,
                ).to(self.device)
                pipe_phi.unet.class_embedding = self.camera_embedding

            if self.cfg.vsd_use_lora:
                # set up LoRA layers
                lora_attn_procs = {}
                for name in pipe_phi.unet.attn_processors.keys():
                    cross_attention_dim = (
                        None
                        if name.endswith("attn1.processor")
                        else pipe_phi.unet.config.cross_attention_dim
                    )
                    if name.startswith("mid_block"):
                        hidden_size = pipe_phi.unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(
                            reversed(pipe_phi.unet.config.block_out_channels)
                        )[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = pipe_phi.unet.config.block_out_channels[block_id]

                    lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                    )

                pipe_phi.unet.set_attn_processor(lora_attn_procs)

                self.lora_layers = AttnProcsLayers(pipe_phi.unet.attn_processors).to(
                    self.device
                )
                self.lora_layers._load_state_dict_pre_hooks.clear()
                self.lora_layers._state_dict_hooks.clear()

        threestudio.info(f"Loaded Stable Diffusion!")

        # controlnet
        controlnet = None
        if self.cfg.controlnet_model_name_or_path is not None:
            threestudio.info(f"Loading ControlNet ...")

            controlnet = ControlNetModel.from_pretrained(
                self.cfg.controlnet_model_name_or_path,
                torch_dtype=self.weights_dtype,
            ).to(self.device)
            controlnet.eval()
            enable_gradient(controlnet, enabled=False)

            threestudio.info(f"Loaded ControlNet!")

        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        # q(z_t|x) = N(alpha_t x, sigma_t^2 I)
        # in DDPM, alpha_t = sqrt(alphas_cumprod_t), sigma_t^2 = 1 - alphas_cumprod_t
        self.alphas_cumprod: Float[Tensor, "T"] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.alphas: Float[Tensor, "T"] = self.alphas_cumprod**0.5
        self.sigmas: Float[Tensor, "T"] = (1 - self.alphas_cumprod) ** 0.5
        # log SNR
        self.lambdas: Float[Tensor, "T"] = self.sigmas / self.alphas

        self._non_trainable_modules = NonTrainableModules(
            pipe=pipe,
            pipe_phi=pipe_phi,
            controlnet=controlnet,
        )

    @property
    def pipe(self) -> StableDiffusionPipeline:
        return self._non_trainable_modules.pipe

    @property
    def pipe_phi(self) -> StableDiffusionPipeline:
        if self._non_trainable_modules.pipe_phi is None:
            raise RuntimeError("phi model is not available.")
        return self._non_trainable_modules.pipe_phi

    @property
    def controlnet(self) -> ControlNetModel:
        if self._non_trainable_modules.controlnet is None:
            raise RuntimeError("ControlNet model is not available.")
        return self._non_trainable_modules.controlnet

    def prepare_pipe(self, pipe: StableDiffusionPipeline):
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
                pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            pipe.unet.to(memory_format=torch.channels_last)

        # FIXME: pipe.__call__ requires text_encoder.dtype
        # pipe.text_encoder.to("meta")
        cleanup()

        pipe.vae.eval()
        pipe.unet.eval()

        enable_gradient(pipe.vae, enabled=False)
        enable_gradient(pipe.unet, enabled=False)

        # disable progress bar
        pipe.set_progress_bar_config(disable=True)

    def configure_pipe_token_merging(self, pipe: StableDiffusionPipeline):
        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(pipe.unet, **self.cfg.token_merging_params)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Int[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "..."]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Float[Tensor, "..."]] = None,
        mid_block_additional_residual: Optional[Float[Tensor, "..."]] = None,
        velocity_to_epsilon: bool = False,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        pred = unet(
            latents.to(unet.dtype),
            t.to(unet.dtype),
            encoder_hidden_states=encoder_hidden_states.to(unet.dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample
        if velocity_to_epsilon:
            pred = latents * self.sigmas[t].view(-1, 1, 1, 1) + pred * self.alphas[
                t
            ].view(-1, 1, 1, 1)
        return pred.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def vae_encode(
        self, vae: AutoencoderKL, imgs: Float[Tensor, "B 3 H W"], mode=False
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        # expect input in [-1, 1]
        input_dtype = imgs.dtype
        posterior = vae.encode(imgs.to(vae.dtype)).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def vae_decode(
        self, vae: AutoencoderKL, latents: Float[Tensor, "B 4 Hl Wl"]
    ) -> Float[Tensor, "B 3 H W"]:
        # output in [0, 1]
        input_dtype = latents.dtype
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents.to(vae.dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    @contextmanager
    def set_scheduler(
        self, pipe: StableDiffusionPipeline, scheduler_class: Any, **kwargs
    ):
        scheduler_orig = pipe.scheduler
        pipe.scheduler = scheduler_class.from_config(scheduler_orig.config, **kwargs)
        yield pipe
        pipe.scheduler = scheduler_orig

    def get_eps_pretrain(
        self,
        latents_noisy: Float[Tensor, "B 4 Hl Wl"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        batch_size = latents_noisy.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                with self.disable_unet_class_embedding(self.pipe.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        torch.cat([latents_noisy] * 4, dim=0),
                        torch.cat([t] * 4, dim=0),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs={"scale": 0.0}
                        if self.vsd_share_model
                        else None,
                        velocity_to_epsilon=self.pipe.scheduler.config.prediction_type
                        == "v_prediction",
                    )  # (4B, 3, Hl, Wl)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                with self.disable_unet_class_embedding(self.pipe.unet) as unet:
                    noise_pred = self.forward_unet(
                        unet,
                        torch.cat([latents_noisy] * 2, dim=0),
                        torch.cat([t] * 2, dim=0),
                        encoder_hidden_states=text_embeddings,
                        cross_attention_kwargs={"scale": 0.0}
                        if self.vsd_share_model
                        else None,
                        velocity_to_epsilon=self.pipe.scheduler.config.prediction_type
                        == "v_prediction",
                    )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    def get_eps_phi(
        self,
        latents_noisy: Float[Tensor, "B 4 Hl Wl"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition: Float[Tensor, "B ..."],
    ) -> Float[Tensor, "B 4 Hl Wl"]:
        batch_size = latents_noisy.shape[0]

        # not using view-dependent prompting in LoRA
        text_embeddings, _ = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        ).chunk(2)
        with torch.no_grad():
            noise_pred = self.forward_unet(
                self.pipe_phi.unet,
                torch.cat([latents_noisy] * 2, dim=0),
                torch.cat([t] * 2, dim=0),
                encoder_hidden_states=torch.cat([text_embeddings] * 2, dim=0),
                class_labels=torch.cat(
                    [
                        camera_condition.view(batch_size, -1),
                        torch.zeros_like(camera_condition.view(batch_size, -1)),
                    ],
                    dim=0,
                )
                if self.cfg.vsd_use_camera_condition
                else None,
                cross_attention_kwargs={"scale": 1.0},
                velocity_to_epsilon=self.pipe_phi.scheduler.config.prediction_type
                == "v_prediction",
            )

        noise_pred_camera, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.vsd_guidance_scale_phi * (
            noise_pred_camera - noise_pred_uncond
        )

        return noise_pred

    def train_phi(
        self,
        latents: Float[Tensor, "B 4 Hl Wl"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_condition: Float[Tensor, "B ..."],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(
            self.cfg.vsd_lora_n_timestamp_samples, 1, 1, 1
        )

        num_train_timesteps = self.pipe_phi.scheduler.config.num_train_timesteps
        t = torch.randint(
            int(num_train_timesteps * 0.0),
            int(num_train_timesteps * 1.0),
            [B * self.cfg.vsd_lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        latents_noisy = self.pipe_phi.scheduler.add_noise(latents, noise, t)
        if self.pipe_phi.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.pipe_phi.scheduler.prediction_type == "v_prediction":
            target = self.pipe_phi.scheduler.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.pipe_phi.scheduler.prediction_type}"
            )

        # not using view-dependent prompting in LoRA
        text_embeddings, _ = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        ).chunk(2)

        if (
            self.cfg.vsd_use_camera_condition
            and self.cfg.vsd_lora_cfg_training
            and random.random() < 0.1
        ):
            camera_condition = torch.zeros_like(camera_condition)

        noise_pred = self.forward_unet(
            self.pipe_phi.unet,
            latents_noisy,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.vsd_lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.vsd_lora_n_timestamp_samples, 1
            )
            if self.cfg.vsd_use_camera_condition
            else None,
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 Hl Wl"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            # treat input rgb as latents
            # input rgb should be in range [-1, 1]
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # treat input rgb as rgb
            # input rgb should be in range [0, 1]
            # encode image into latents with vae
            latents = self.vae_encode(self.pipe.vae, rgb_BCHW_512 * 2.0 - 1.0)

        # sample timestep
        # use the same timestep for each batch
        assert self.min_step is not None and self.max_step is not None
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [1],
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)

        # sample noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        eps_pretrain = self.get_eps_pretrain(
            latents_noisy, t, prompt_utils, elevation, azimuth, camera_distances
        )

        latents_1step_orig = (
            1
            / self.alphas[t].view(-1, 1, 1, 1)
            * (latents_noisy - self.sigmas[t].view(-1, 1, 1, 1) * eps_pretrain)
        ).detach()

        if self.cfg.guidance_type == "sds":
            eps_phi = noise
        elif self.cfg.guidance_type == "vsd":
            if self.cfg.vsd_camera_condition_type == "extrinsics":
                camera_condition = c2w
            elif self.cfg.vsd_camera_condition_type == "mvp":
                camera_condition = mvp_mtx
            elif self.cfg.vsd_camera_condition_type == "spherical":
                camera_condition = torch.stack(
                    [
                        torch.deg2rad(elevation),
                        torch.sin(torch.deg2rad(azimuth)),
                        torch.cos(torch.deg2rad(azimuth)),
                        camera_distances,
                    ],
                    dim=-1,
                )
            else:
                raise ValueError(
                    f"Unknown camera_condition_type {self.cfg.vsd_camera_condition_type}"
                )
            eps_phi = self.get_eps_phi(
                latents_noisy,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                camera_condition,
            )

            loss_train_phi = self.train_phi(
                latents,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                camera_condition,
            )

        if self.cfg.weighting_strategy == "dreamfusion":
            w = (1.0 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1.0
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (eps_pretrain - eps_phi)

        # compute decoded image if needed for visualization/img loss
        if self.cfg.return_rgb_1step_orig or self.cfg.use_img_loss:
            with torch.no_grad():
                image_denoised_pretrain = self.vae_decode(
                    self.pipe.vae, latents_1step_orig
                )
                rgb_1step_orig = image_denoised_pretrain.permute(0, 2, 3, 1)

        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick:
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_sd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sd": loss_sd,
            "grad_norm": grad.norm(),
            "timesteps": t,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "latents": latents,
            "latents_1step_orig": latents_1step_orig,
            "rgb": rgb_BCHW.permute(0, 2, 3, 1),
            "weights": w,
            "lambdas": self.lambdas[t],
        }

        # image-space loss proposed in HiFA: https://hifa-team.github.io/HiFA-site
        if self.cfg.use_img_loss:
            if self.cfg.guidance_type == "vsd":
                latents_denoised_est = (
                    latents_noisy - self.sigmas[t] * eps_phi
                ) / self.alphas[t].view(-1, 1, 1, 1)
                image_denoised_est = self.vae_decode(
                    self.pipe.vae, latents_denoised_est
                )
            else:
                image_denoised_est = rgb_BCHW_512
            grad_img = (
                w
                * (image_denoised_est - image_denoised_pretrain)
                * self.alphas[t].view(-1, 1, 1, 1)
                / self.sigmas[t].view(-1, 1, 1, 1)
            )
            if self.grad_clip_val is not None:
                grad_img = grad_img.clamp(-self.grad_clip_val, self.grad_clip_val)
            target_img = (rgb_BCHW_512 - grad_img).detach()
            loss_sd_img = (
                0.5 * F.mse_loss(rgb_BCHW_512, target_img, reduction="sum") / batch_size
            )
            guidance_out.update({"loss_sd_img": loss_sd_img})

        if self.cfg.return_rgb_1step_orig:
            guidance_out.update({"rgb_1step_orig": rgb_1step_orig})

        if self.cfg.return_rgb_multistep_orig:
            with self.set_scheduler(
                self.pipe,
                DPMSolverSinglestepScheduler,
                solver_order=1,
                num_train_timesteps=int(t[0]),
            ) as pipe:
                text_embeddings = prompt_utils.get_text_embeddings(
                    elevation,
                    azimuth,
                    camera_distances,
                    self.cfg.view_dependent_prompting,
                )
                text_embeddings_cond, text_embeddings_uncond = text_embeddings.chunk(2)
                with torch.cuda.amp.autocast(enabled=False):
                    latents_multistep_orig = pipe(
                        num_inference_steps=self.cfg.n_rgb_multistep_orig_steps,
                        guidance_scale=self.cfg.guidance_scale,
                        eta=1.0,
                        latents=latents_noisy.to(pipe.unet.dtype),
                        prompt_embeds=text_embeddings_cond.to(pipe.unet.dtype),
                        negative_prompt_embeds=text_embeddings_uncond.to(
                            pipe.unet.dtype
                        ),
                        cross_attention_kwargs={"scale": 0.0}
                        if self.vsd_share_model
                        else None,
                        output_type="latent",
                    ).images.to(latents.dtype)
            with torch.no_grad():
                rgb_multistep_orig = self.vae_decode(
                    self.pipe.vae, latents_multistep_orig
                )
            guidance_out.update(
                {
                    "latents_multistep_orig": latents_multistep_orig,
                    "rgb_multistep_orig": rgb_multistep_orig.permute(0, 2, 3, 1),
                }
            )

        if self.cfg.guidance_type == "vsd":
            guidance_out.update(
                {
                    "loss_train_phi": loss_train_phi,
                }
            )

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.sqrt_anneal:
            percentage = (
                float(global_step) / self.cfg.trainer_max_steps
            ) ** 0.5  # progress percentage
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
