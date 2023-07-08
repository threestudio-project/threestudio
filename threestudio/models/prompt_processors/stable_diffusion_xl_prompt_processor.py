import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import threestudio
from threestudio.models.prompt_processors.base import (
    PromptProcessor,
    PromptProcessorOutput,
    hash_prompt,
)
from threestudio.utils.misc import barrier, cleanup
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-xl-prompt-processor")
class StableDiffusionXLPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings, self.text_embeddings_pooled = self.load_from_cache(
            self.prompt
        )
        self.text_embeddings, self.text_embeddings_pooled = (
            self.text_embeddings[None, ...],
            self.text_embeddings_pooled[None, ...],
        )
        (
            self.uncond_text_embeddings,
            self.uncond_text_embeddings_pooled,
        ) = self.load_from_cache(self.negative_prompt)
        self.uncond_text_embeddings, self.uncond_text_embeddings_pooled = (
            self.uncond_text_embeddings[None, ...],
            self.uncond_text_embeddings_pooled[None, ...],
        )

        embeds_vd_cache = [self.load_from_cache(prompt) for prompt in self.prompts_vd]
        self.text_embeddings_vd = torch.stack([e[0] for e in embeds_vd_cache], dim=0)
        self.text_embeddings_pooled_vd = torch.stack(
            [e[1] for e in embeds_vd_cache], dim=0
        )
        uncond_embeds_vd_cache = [
            self.load_from_cache(prompt) for prompt in self.negative_prompts_vd
        ]
        self.uncond_text_embeddings_vd = torch.stack(
            [e[0] for e in uncond_embeds_vd_cache], dim=0
        )
        self.uncond_text_embeddings_pooled_vd = torch.stack(
            [e[1] for e in uncond_embeds_vd_cache], dim=0
        )

        threestudio.debug(f"Loaded text embeddings.")

    def __call__(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
            # pooled text embeddings
            text_embeddings_pooled=self.text_embeddings_pooled,
            uncond_text_embeddings_pooled=self.uncond_text_embeddings_pooled,
            text_embeddings_pooled_vd=self.text_embeddings_pooled_vd,
            uncond_text_embeddings_pooled_vd=self.uncond_text_embeddings_pooled_vd,
        )

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            device_map="auto",
        )

        tokenizers = [tokenizer, tokenizer_2]
        text_encoders = [text_encoder, text_encoder_2]

        with torch.no_grad():
            text_embeddings_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                tokens = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_encoder_out = text_encoder(
                    tokens.input_ids,
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the last text encoder
                text_embeddings_pooled = text_encoder_out[0]
                text_embeddings_list.append(text_encoder_out.hidden_states[-2])
            text_embeddings = torch.cat(text_embeddings_list, dim=-1)

        for prompt, embedding, embedding_pooled in zip(
            prompts, text_embeddings, text_embeddings_pooled
        ):
            torch.save(
                (embedding, embedding_pooled),
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
        del text_encoder_2
