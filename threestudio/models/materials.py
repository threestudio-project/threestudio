from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.base import BaseModule
from threestudio.utils.ops import get_activation, dot
from threestudio.utils.typing import *


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        raise NotImplementedError


@threestudio.register('neural-radiance-material')
class NeuralRadianceMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 8
        color_activation: str = 'sigmoid'
        dir_encoding_config: dict = field(default_factory=lambda: {
            'otype': "SphericalHarmonics",
            'degree': 3
        })
        mlp_network_config: dict = field(default_factory=lambda: {
            'otype': "FullyFusedMLP",
            'activation': "ReLU",
            'n_neurons': 16,
            'n_hidden_layers': 2
        })

    cfg: Config
    
    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.n_input_dims = self.cfg.input_feature_dims + self.encoding.n_output_dims # type: ignore
        self.network = get_mlp(self.n_input_dims, 3, self.cfg.mlp_network_config)    
    
    def forward(self, features: Float[Tensor, "*B Nf"], viewdirs: Float[Tensor, "*B 3"], **kwargs) -> Float[Tensor, "*B 3"]:
        # viewdirs and normals must be normalized before passing to this function
        viewdirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        viewdirs_embd = self.encoding(viewdirs.view(-1, 3))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), viewdirs_embd], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], 3).float()
        color = get_activation(self.cfg.color_activation)(color)
        return color


@threestudio.register('diffuse-with-point-light-material')
class DiffuseWithPointLightMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
        ambient_only_steps: int = 1000
        diffuse_prob: float = 0.75
        textureless_prob: float = 0.5
        albedo_activation: str = 'sigmoid'

    cfg: Config

    def configure(self) -> None:
        self.ambient_light_color: Float[Tensor, "3"]
        self.register_buffer('ambient_light_color', torch.as_tensor(self.cfg.ambient_light_color, dtype=torch.float32))
        self.diffuse_light_color: Float[Tensor, "3"]
        self.register_buffer('diffuse_light_color', torch.as_tensor(self.cfg.diffuse_light_color, dtype=torch.float32))
        self.ambient_only = False

    @typechecker
    def forward(self, features: Float[Tensor, "B ... Nf"], positions: Float[Tensor, "B ... 3"], shading_normal: Float[Tensor, "B ... 3"], light_positions: Float[Tensor, "B ... 3"], **kwargs) -> Float[Tensor, "B ... 3"]:
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3])
        if self.ambient_only:
            return albedo
        light_directions: Float[Tensor, "B ... 3"] = F.normalize(light_positions - positions, dim=-1)
        diffuse_light: Float[Tensor, "B ... 3"] = dot(shading_normal, light_directions).clamp(min=0.) * self.ambient_light_color
        textureless_color = diffuse_light + self.ambient_light_color
        color = albedo * textureless_color

        # adopt the same type of augmentation for the whole batch
        if torch.rand([]) > self.cfg.diffuse_prob:
            return albedo
        if torch.rand([]) < self.cfg.textureless_prob:
            return textureless_color
        return color

    def update_step(self, epoch: int, global_step: int):
        if global_step < self.cfg.ambient_only_steps:
            self.ambient_only = True
        else:
            self.ambient_only = False
