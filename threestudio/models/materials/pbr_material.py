import random
from dataclasses import dataclass, field

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("pbr-material")
class PBRMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 32
        material_activation: str = "sigmoid"
        light_activation: str = "exp"
        material_mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        diffuse_mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        specular_mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        dir_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "IntegratedDirectionalEncoding",
                "degree": 5,
            }
        )

    cfg: Config

    def configure(self) -> None:
        self.material_network = get_mlp(
            self.cfg.input_feature_dims, 3 + 1 + 1, self.cfg.material_mlp_network_config
        )
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.diffuse_light_network = get_mlp(
            self.encoding.n_output_dims, 3, self.cfg.diffuse_mlp_network_config
        )
        self.specular_light_network = get_mlp(
            self.encoding.n_output_dims, 3, self.cfg.specular_mlp_network_config
        )

        FG_LUT = torch.from_numpy(
            np.fromfile("load/bsdf/bsdf_256_256.bin", dtype=np.float32).reshape(
                1, 256, 256, 2
            )
        )
        self.register_buffer("FG_LUT", FG_LUT)

    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        prefix_shape = features.shape[:-1]

        # viewdirs and normals must be normalized before passing to this function
        wo = -viewdirs
        normal_dot_wo = (shading_normal * wo).sum(-1, keepdim=True)
        reflective = normal_dot_wo * shading_normal * 2 - wo

        material = self.material_network(features.view(-1, features.shape[-1])).view(
            *prefix_shape, -1
        )
        material = get_activation(self.cfg.material_activation)(material)
        albedo = material[..., :3]
        metallic = material[..., 3:4]
        roughness = material[..., 4:5]

        diffuse_albedo = (1 - metallic) * albedo

        fg_uv = torch.cat([normal_dot_wo, roughness], -1).clamp(
            0, 1
        )  # [*prefix_shape, 2]
        fg = dr.texture(
            self.FG_LUT,
            fg_uv.reshape(1, -1, 1, 2).contiguous(),
            filter_mode="linear",
            boundary_mode="clamp",
        ).reshape(*prefix_shape, 2)
        specular_albedo = (0.04 * (1 - metallic) + metallic * albedo) * fg[:, 0:1] + fg[
            :, 1:2
        ]

        diffuse_light = self.diffuse_light_network(self.encoding(shading_normal))
        diffuse_light = get_activation(self.cfg.light_activation)(
            diffuse_light.clamp(max=5)
        )

        specular_light = self.specular_light_network(
            self.encoding(reflective, roughness)
        )
        specular_light = get_activation(self.cfg.light_activation)(
            specular_light.clamp(max=5)
        )

        color = diffuse_albedo * diffuse_light + specular_albedo * specular_light
        color = color.clamp(0.0, 1.0)

        return color

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        prefix_shape = features.shape[:-1]

        material = self.material_network(features.view(-1, features.shape[-1])).view(
            *prefix_shape, -1
        )
        material = get_activation(self.cfg.material_activation)(material)
        albedo = material[..., :3]
        metallic = material[..., 3:4]
        roughness = material[..., 4:5]

        return {
            "albedo": albedo,
            "metallic": metallic,
            "roughness": roughness,
        }
