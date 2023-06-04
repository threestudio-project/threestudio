# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType
from tqdm import tqdm
from models.controlnet.annotator.normalbae import NormalBaeDetector


CONSOLE = Console(width=120)

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    DDIMScheduler

)
from transformers import logging


logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
SD_SOURCE = 'SG161222/Realistic_Vision_V2.0'
# SD_SOURCE = "nitrosocke/nitro-diffusion"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class ControlNet(nn.Module):
    """ControlNet implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, use_full_precision=False, control_type: str = 'normalbae') -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.use_full_precision = use_full_precision

        self.control_type = control_type
        if self.control_type == 'normalbae':
            model_source = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.control_type == 'ip2p':
            model_source = "lllyasviel/control_v11e_sd15_ip2p"
        elif self.control_type ==  'depth':
            model_source = "lllyasviel/control_v11f1p_sd15_depth"
        elif self.control_type ==  'openpose':
            model_source = "lllyasviel/control_v11p_sd15_openpose"
        else:
            raise NotImplementedError(f"{self.control_type} is not supprted!")

        controlnet = ControlNetModel.from_pretrained(model_source, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_SOURCE, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, cache_dir='/root/autodl-tmp/cache')
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler", cache_dir='/root/autodl-tmp/cache') # UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()
        pipe.controlnet.eval()

        # use for improved quality at cost of higher memory
        if self.use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
            pipe.controlnet.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae
        self.controlnet = pipe.controlnet

        if self.control_type == 'normalbae':
            self.preprocessor = NormalBaeDetector()

        CONSOLE.print("ControNet loaded!")

    def edit_image(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_origin: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        print(T)
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")
        print("scheduler_is_in_sigma_space:", scheduler_is_in_sigma_space)
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond = self.prepare_control_image(image_origin)

        # add noise
        # torch.manual_seed(12345)
        print(self.scheduler.timesteps[0])
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 2)

                # controlnet(s) inference
                controlnet_latent_model_input = latent_model_input
                controlnet_prompt_embeds = text_embeddings

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image_cond,
                    conditioning_scale=conditioning_scale,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=None, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,).sample

            # perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img, T.cpu()

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def prepare_control_image(self, image):
        if self.control_type  == 'ip2p':
            return image
        else:
            assert self.preprocessor is not None
            return self.preprocessor(image)

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
