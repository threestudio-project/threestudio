import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.typing import *


@threestudio.register("sd-latent-adapter-material")
class StableDiffusionLatentAdapterMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        pass

    cfg: Config

    def configure(self) -> None:
        adapter = nn.Parameter(
            torch.as_tensor(
                [
                    #   R       G       B
                    [0.298, 0.207, 0.208],  # L1
                    [0.187, 0.286, 0.173],  # L2
                    [-0.158, 0.189, 0.264],  # L3
                    [-0.184, -0.271, -0.473],  # L4
                ]
            )
        )
        self.register_parameter("adapter", adapter)

    def forward(
        self, features: Float[Tensor, "B ... 4"], **kwargs
    ) -> Float[Tensor, "B ... 3"]:
        assert features.shape[-1] == 4
        color = features @ self.adapter
        color = (color + 1) / 2
        color = color.clamp(0.0, 1.0)
        return color
