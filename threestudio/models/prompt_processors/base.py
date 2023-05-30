import json
import os
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import barrier, cleanup, get_rank
from threestudio.utils.ops import shifted_expotional_decay, shifted_cosine_decay
from threestudio.utils.typing import *


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


class PromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt: str = "a hamburger"
        negative_prompt: str = ""
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        view_dependent_prompting: bool = True
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0
        view_dependent_prompt_front: bool = False
        use_cache: bool = True
        spawn: bool = True

        # for perp-neg
        use_perp_neg: bool = False
        # a*e(-b*r) + c
        # a * e(-b) + c = 0
        #
        # f_sb: Tuple[float, float, float] = (4, 0.5, -2.426)
        # f_fsb: Tuple[float, float, float] = (4, 0.5, -2.426)
        # f_fs: Tuple[float, float, float] = (4, 0.5, -2.426)  # f_fs(1) = 0, a, b > 0
        # f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

        f_sb: Tuple[float, float, float] = (4, 0.5, -2.426)
        f_fsb: Tuple[float, float, float] = (4, 0.5, -2.426)
        f_fs: Tuple[float, float, float] = (4, 0.5, -2.426)  # f_fs(1) = 0, a, b > 0
        f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

    cfg: Config

    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        @dataclass
        class DirectionConfig:
            name: str
            prompt: Callable[[str], str]
            negative_prompt: Callable[[str], str]
            condition: Callable[
                [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
                Float[Tensor, "B"],
            ]

        # view-dependent text embeddings
        self.directions: List[DirectionConfig]
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > -self.cfg.front_threshold)
                    & (azi < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > 180 - self.cfg.back_threshold)
                    | (azi < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > -self.cfg.front_threshold)
                    & (azi < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (azi > 180 - self.cfg.back_threshold)
                    | (azi < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        with open(os.path.join("load/prompt_library.json"), "r") as f:
            self.prompt_library = json.load(f)
        # use provided prompt or find prompt in library
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt
        self.prompts_vd = [d.prompt(self.prompt) for d in self.directions]
        self.negative_prompts_vd = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        self.prepare_text_embeddings()
        self.load_text_embeddings()

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        raise NotImplementedError

    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.prompt]
            + [self.negative_prompt]
            + self.prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            if self.cfg.spawn:
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )
                subprocess.start()
                subprocess.join()
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                )
            cleanup()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    def preprocess_prompt(self, prompt: str) -> str:
        if prompt.startswith("lib:"):
            # find matches in the library
            candidate = None
            keywords = prompt[4:].lower().split("_")
            for prompt in self.prompt_library["dreamfusion"]:
                if all([k in prompt.lower() for k in keywords]):
                    if candidate is not None:
                        raise ValueError(
                            f"Multiple prompts matched with keywords {keywords} in library"
                        )
                    candidate = prompt
            if candidate is None:
                raise ValueError(
                    f"Cannot find prompt with keywords {keywords} in library"
                )
            return candidate
        else:
            return prompt

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError

    def __call__(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        **kwargs,
    ) -> Dict[str, Any]:
        res = {}

        if self.cfg.view_dependent_prompting:
            if self.cfg.use_perp_neg:
                (
                    uncond_text_embeddings,
                    pos_text_embeddings,
                    neg_text_embeddings,
                    neg_guidance_weights,
                ) = self.get_perp_neg_view_dependent_text_embeddings(
                    elevation, azimuth, camera_distances
                )
                res.update(
                    {
                        "uncond_text_embeddings": uncond_text_embeddings,
                        "pos_text_embeddings": pos_text_embeddings,
                        "neg_text_embeddings": neg_text_embeddings,
                        "neg_guidance_weights": neg_guidance_weights,
                    }
                )
            else:
                (
                    text_embeddings,
                    uncond_text_embeddings,
                ) = self.get_vanilla_view_dependent_text_embeddings(
                    elevation, azimuth, camera_distances
                )
                res.update(
                    {
                        "text_embeddings": torch.cat(
                            [text_embeddings, uncond_text_embeddings], dim=0
                        )
                    }
                )
        else:
            batch_size = elevation.shape[0]
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

            # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
            res.update(
                {
                    "text_embeddings": torch.cat(
                        [text_embeddings, uncond_text_embeddings], dim=0
                    )
                }
            )
        return res

    def get_vanilla_view_dependent_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        # Get direction
        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]

        # Get text embeddings
        text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
        uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        return text_embeddings, uncond_text_embeddings

    def get_perp_neg_view_dependent_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ) -> Tuple[
        Float[Tensor, "B ..."],
        Float[Tensor, "B ..."],
        Float[Tensor, "B 2 ..."],
        Float[Tensor, "B 2"],
    ]:
        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []

        side_emb = self.text_embeddings_vd[0]
        front_emb = self.text_embeddings_vd[1]
        back_emb = self.text_embeddings_vd[2]
        overhead_emb = self.text_embeddings_vd[3]

        for idx, ele, azi, dis in zip(
            direction_idx, elevation, azimuth, camera_distances
        ):
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete front, 1 - complete side
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.cfg.f_fs, r_inter),
                        -shifted_expotional_decay(*self.cfg.f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.cfg.f_sb, r_inter),
                        -shifted_expotional_decay(*self.cfg.f_fsb, r_inter),
                    ]
        uncond_text_embeddings = torch.stack(uncond_text_embeddings, dim=0)
        pos_text_embeddings = torch.stack(pos_text_embeddings, dim=0)
        neg_text_embeddings = torch.stack(neg_text_embeddings, dim=0).reshape(
            batch_size, -1, *uncond_text_embeddings.shape[1:]
        )
        neg_guidance_weights = torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, -1)
        # breakpoint()
        return (
            uncond_text_embeddings,
            pos_text_embeddings,
            neg_text_embeddings,
            neg_guidance_weights,
        )
