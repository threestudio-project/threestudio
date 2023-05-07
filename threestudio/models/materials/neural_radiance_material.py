import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("neural-radiance-material")
class NeuralRadianceMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 8
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.n_input_dims = self.cfg.input_feature_dims + self.encoding.n_output_dims  # type: ignore
        self.network = get_mlp(self.n_input_dims, 3, self.cfg.mlp_network_config)

    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        # viewdirs and normals must be normalized before passing to this function
        viewdirs = (viewdirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        viewdirs_embd = self.encoding(viewdirs.view(-1, 3))
        network_inp = torch.cat(
            [features.view(-1, features.shape[-1]), viewdirs_embd], dim=-1
        )
        color = self.network(network_inp).view(*features.shape[:-1], 3)
        color = get_activation(self.cfg.color_activation)(color)
        return color
