import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import IFPipeline
from transformers import T5EncoderModel, T5Tokenizer

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("deep-floyd-prompt-processor")
class DeepFloydPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )  # FIXME: behavior of auto device map in multi-GPU training
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

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        max_length = 77
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,  # suppress warning
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )
        with torch.no_grad():
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            text_embeddings = text_encoder(
                text_input_ids.to(text_encoder.device),
                attention_mask=attention_mask.to(text_encoder.device),
            )
            text_embeddings = text_embeddings[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
