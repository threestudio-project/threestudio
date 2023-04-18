import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class PromptProcessor(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError



@threestudio.register('dreamfusion-prompt-processor')
class DreamFusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        prompt: str = "a hamburger"
        negative_prompt: str = ""
        pretrained_model_name_or_path: str = 'runwayml/stable-diffusion-v1-5'
        view_dependent_prompting: bool = True
        overhead_threshold: float = 60.
        front_threshold: float = 22.5
        back_threshold: float = 22.5

    cfg: Config

    def configure(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder='tokenizer')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.text_encoder = CLIPTextModel.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder='text_encoder').to(self.device)
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        @dataclass
        class DirectionConfig:
            name: str
            prompt: str
            negative_prompt: str
            condition: Callable[[Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]], Float[Tensor, "B"]]
        self.directions: List[DirectionConfig] = [
            DirectionConfig('side', ', side view', '', lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool)),
            DirectionConfig('front', ', front view', '', lambda ele, azi, dis: (azi > -self.cfg.front_threshold) & (azi < self.cfg.front_threshold)),
            DirectionConfig('back', ', back view', '', lambda ele, azi, dis: (azi > 180 - self.cfg.back_threshold) | (azi < -180 + self.cfg.back_threshold)),
            DirectionConfig('overhead', ', overhead view', '', lambda ele, azi, dis: ele > self.cfg.overhead_threshold)
        ]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}
        self.text_embeddings, self.uncond_text_embeddings = self.get_text_embeddings(
            [self.cfg.prompt], [self.cfg.negative_prompt]
        )
        # view-dependent text embeddings
        self.text_embeddings_vd, self.uncond_text_embeddings_vd = self.get_text_embeddings(
            [f"{self.cfg.prompt} {d.prompt}" for d in self.directions],
            [f"{self.cfg.negative_prompt} {d.negative_prompt}" for d in self.directions],
        )

    def get_text_embeddings(self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(uncond_tokens.input_ids.to(self.device))[0]

        return text_embeddings, uncond_text_embeddings

    def forward(self, elevation: Float[Tensor, "B"], azimuth: Float[Tensor, "B"], camera_distances: Float[Tensor, "B"], **kwargs) -> Float[Tensor, "BB 77 768"]:
        batch_size = elevation.shape[0]

        if self.cfg.view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[d.condition(elevation, azimuth, camera_distances)] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)
            uncond_text_embeddings = self.uncond_text_embeddings.expand(batch_size, -1, -1)

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
        
