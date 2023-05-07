import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import IFPipeline
from transformers import T5EncoderModel

import threestudio
from threestudio.models.prompt.base import PromptProcessor
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("deep-floyd-prompt-processor")
class DeepFloydPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"

    cfg: Config

    def configure_text_encoder(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # FIXME: behavior of auto device map
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,  # pass the previously instantiated 8bit text encoder
            unet=None,
        )

    def destroy_text_encoder(self) -> None:
        del self.text_encoder
        del self.pipe
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 4096"], Float[Tensor, "B 77 4096"]]:
        text_embeddings, uncond_text_embeddings = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device=self.device
        )
        return text_embeddings, uncond_text_embeddings
