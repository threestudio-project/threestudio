from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *


@threestudio.register("solid-rgb-feature-background")
class SolidRgbFeatureBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: float = 0.0
        feature: float = 0.0
        learned: bool = False

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        color = [self.cfg.color for i in range(3)]
        color += [self.cfg.feature for i in range(self.cfg.n_output_dims - 3)]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(color, dtype=torch.float32)
            )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        return (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
            * self.env_color
        )
