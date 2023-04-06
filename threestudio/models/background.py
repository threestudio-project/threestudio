from dataclasses import dataclass, field

import torch
import torch.nn as nn

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
        color: Tuple[float, float, float] = (1., 1., 1.)

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "3"]
        self.register_buffer('env_color', torch.as_tensor(self.cfg.color, dtype=torch.float32))

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        return torch.ones_like(dirs) * self.env_color


@threestudio.register("neural-environment-map-background")
class NeuralEnvironmentMapBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        color_activation: str = 'sigmoid'
        dir_encoding_config: dict = field(default_factory=lambda: {
            "otype": "SphericalHarmonics",
            "degree": 3
        })
        mlp_network_config: dict = field(default_factory=lambda: {
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "n_neurons": 16,
            "n_hidden_layers": 2
        })

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.network = get_mlp(self.encoding.n_output_dims, 3, self.cfg.mlp_network_config)

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B 3"]:
        # viewdirs must be normalized before passing to this function
        dirs = (dirs + 1.) / 2.  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, 3))
        color = self.network(dirs_embd).view(*dirs.shape[:-1], 3).float()
        color = get_activation(self.cfg.color_activation)(color)
        return color
