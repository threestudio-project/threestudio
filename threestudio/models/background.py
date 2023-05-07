from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.base import BaseModule
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        raise NotImplementedError


@threestudio.register("solid-color-background")
class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32)
            )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        return (
            torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs)
            * self.env_color
        )


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
        azimuth = torch.atan2(y, x) / (torch.pi * 2) + 0.5
        elevation = torch.atan2(z, xy) / torch.pi + 0.5
        uv = torch.stack([azimuth, elevation], -1)
        return uv

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        dirs_shape = dirs.shape[:-1]
        uv = self.spherical_xyz_to_uv(dirs)
        uv = 2 * uv - 1  # rescale to [-1, 1] for grid_sample
        uv = uv.reshape(1, -1, 1, 2)
        color = (
            F.grid_sample(self.texture, uv)
            .reshape(self.cfg.n_output_dims, -1)
            .T.reshape(*dirs_shape, self.cfg.n_output_dims)
        )
        color = get_activation(self.cfg.color_activation)(color)
        return color


@threestudio.register("neural-environment-map-background")
class NeuralEnvironmentMapBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_output_dims,
            self.cfg.mlp_network_config,
        )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        # viewdirs must be normalized before passing to this function
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, 3))
        color = self.network(dirs_embd).view(*dirs.shape[:-1], self.cfg.n_output_dims)
        color = get_activation(self.cfg.color_activation)(color)
        return color
