from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("dynamic-implicit-volume")
class DynamicImplicitVolume(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 4
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        time_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "ProgressiveBandFrequency",
                "n_frequencies": 4,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        
    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pos_encoding = get_encoding(
            self.cfg.n_input_dims - 1, self.cfg.pos_encoding_config
        )
        self.time_encoding = get_encoding(
            1, self.cfg.time_encoding_config
        )
        self.flow_network = get_mlp(
            self.pos_encoding.n_output_dims + self.time_encoding.n_output_dims, 
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config
        )

    def forward(
        self, points_4d: Float[Tensor, "*N Di"]
    ) -> Dict[str, Float[Tensor, "..."]]:

        # points 4D
        points_unscaled = points_4d[..., :self.cfg.n_input_dims-1]  # points in the original scale
        points = contract_to_unisphere(
            points_4d[..., :self.cfg.n_input_dims-1], self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        points_time = points_4d[..., -1:]

        pos_enc = self.pos_encoding(points.view(-1, self.cfg.n_input_dims-1))
        time_enc = self.time_encoding(points_time.view(-1, 1))
        enc = torch.cat([pos_enc, time_enc], dim=-1)
        features = self.flow_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        output = {}
        output.update({"features": features})

        return output

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "DynamicImplicitVolume":
        if isinstance(other, DynamicImplicitVolume):
            instance = DynamicImplicitVolume(cfg, **kwargs)
            instance.pos_encoding.load_state_dict(other.pos_encoding.state_dict())
            instance.time_encoding.load_state_dict(other.time_encoding.state_dict())
            instance.flow_network.load_state_dict(other.flow_network.state_dict())
            return instance
        else:
            raise TypeError(
                f"Cannot create {DynamicImplicitVolume.__name__} from {other.__class__.__name__}"
            )
