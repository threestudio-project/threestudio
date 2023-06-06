import cv2
import importlib
import numpy as np
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("zero123-vsd-guidance")
class Zero123VSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "load/zero123/105000.ckpt"
        pretrained_config: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-1"

        vram_O: bool = True
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        guidance_scale: float = 3.0
        guidance_scale_lora: float = 1.0
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: int = 5000

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zero123 + StableDiffusion...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        # self.weights_dtype = (
        #     torch.float16 if self.cfg.half_precision_weights else torch.float32
        # )
        # TODO: seems Zero123 cannot load into fp16...
        self.weights_dtype = torch.float32

        pipe_lora_kwargs = {
            "tokenizer": None,
            "text_encoder": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            # pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        self.single_model = False
        pipe_lora = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            **pipe_lora_kwargs,
        ).to(self.device)
        # del pipe_lora.vae
        # cleanup()
        # pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe_lora=pipe_lora)

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
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        )
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        # self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
        #     self.pipe.scheduler.config
        # )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        # self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        # self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.prepare_embeddings(self.cfg.cond_image_path)

        threestudio.info(f"Loaded Zero123 + Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: str) -> Float[Tensor, "B 3 256 256"]:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn, self.c_concat = self.get_img_embeds(self.rgb_256)

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c_crossattn=None,
        c_concat=None,
        **kwargs,
    ) -> dict:
        T = torch.stack(
            [
                torch.deg2rad(
                    (90 - elevation) - (90 - self.cfg.cond_elevation_deg)
                ),  # Zero123 polar is 90-elevation
                torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                camera_distances - self.cfg.cond_camera_distance,
            ],
            dim=-1,
        )[:, None, :].to(self.device)
        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (self.c_crossattn if c_crossattn is None else c_crossattn).repeat(
                        len(T), 1, 1
                    ),
                    T,
                ],
                dim=-1,
            )
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat)
                    .repeat(len(T), 1, 1, 1)
                    .to(self.device),
                    (self.c_concat if c_concat is None else c_concat).repeat(
                        len(T), 1, 1, 1
                    ),
                ],
                dim=0,
            )
        ]
        return cond

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        cond: dict,
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
        guidance_eval: bool,
    ):
        B = latents.shape[0]

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred_pretrain = self.model.apply_model(x_in, t_in, cond)

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                x_in,
                t_in,
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # # TODO: more general cases
        # assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = x_in * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)

        if guidance_eval:
            guidance_eval_out = self.evaluate_guidance(
                cond, t, latents_noisy, noise_pred_pretrain
            )
        else:
            guidance_eval_out = {}

        return grad, guidance_eval_out

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.cfg.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 32 32"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (32, 32), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_256 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_256)
        return latents

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
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        cond = self.get_cond(elevation, azimuth, camera_distances)

        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        grad, guidance_eval_out = self.compute_grad_vsd(
            latents, cond, text_embeddings, camera_condition, guidance_eval
        )

        grad = torch.nan_to_num(grad)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        loss_lora = self.train_lora(latents, text_embeddings, camera_condition)

        guidance_out = {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
        }

        return guidance_out, guidance_eval_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if global_step > self.cfg.anneal_start_step:
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def evaluate_guidance(self, cond, t_orig, latents, noise_pred):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents.shape[0]  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(
            [bs, -1]
        ) > t_orig.unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(len(t)):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents[b : b + 1], eta=1
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
            c = {
                "c_crossattn": [cond["c_crossattn"][0][b * 2 : b * 2 + 2]],
                "c_concat": [cond["c_concat"][0][b * 2 : b * 2 + 2]],
            }
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                x_in = torch.cat([latents] * 2)
                t_in = torch.cat([t.reshape(1)] * 2).to(self.device)
                noise_pred = self.model.apply_model(x_in, t_in, c)
                # perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
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
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample_lora(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images_sd(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents_sd(
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

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    # verification - requires `vram_O = False` in load_model_from_config
    @torch.no_grad()
    def sample(
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
