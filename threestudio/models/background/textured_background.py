from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("textured-background")
class TexturedBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        height: int = 64
        width: int = 64
        color_activation: str = "sigmoid"

    cfg: Config

    def configure(self) -> None:
        self.texture = nn.Parameter(
            torch.randn((1, self.cfg.n_output_dims, self.cfg.height, self.cfg.width))
        )

    def spherical_xyz_to_uv(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 2"]:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        xy = (x**2 + y**2) ** 0.5
        u = torch.atan2(xy, z) / torch.pi
        v = torch.atan2(y, x) / (torch.pi * 2) + 0.5
        uv = torch.stack([u, v], -1)
        return uv

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        dirs_shape = dirs.shape[:-1]
        uv = self.spherical_xyz_to_uv(dirs.reshape(-1, dirs.shape[-1]))
        uv = 2 * uv - 1  # rescale to [-1, 1] for grid_sample
        uv = uv.reshape(1, -1, 1, 2)
        color = (
            F.grid_sample(
                self.texture,
                uv,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
            .reshape(self.cfg.n_output_dims, -1)
            .T.reshape(*dirs_shape, self.cfg.n_output_dims)
        )
        color = get_activation(self.cfg.color_activation)(color)
        return color
