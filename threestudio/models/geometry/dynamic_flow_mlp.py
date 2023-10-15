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


@threestudio.register("dynamic-flow-mlp")
class DynamicFlowMLP(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 4
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "ProgressiveBandFrequency",
                "n_frequencies": 6
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
                "n_neurons": 256,
                "n_hidden_layers": 4,
            }
        )
        need_normalization: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.pos_encoding = get_encoding(
            self.cfg.n_input_dims - 1, self.cfg.pos_encoding_config
        )
        self.time_encoding = get_encoding(1, self.cfg.time_encoding_config)
        self.flow_network = get_mlp(
            self.pos_encoding.n_output_dims + self.time_encoding.n_output_dims,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )

    def forward(
        self, points_3d: Float[Tensor, "*N Di"], moment
    ) -> Dict[str, Float[Tensor, "..."]]:
        # points 4D
        if self.cfg.need_normalization:
            points = contract_to_unisphere(
                points_3d[..., : self.cfg.n_input_dims - 1], self.bbox, self.unbounded
            )  # points normalized to (0, 1)
        else:
            points = points_3d[..., : self.cfg.n_input_dims - 1]
        points_time = torch.zeros_like(points[..., 0])
        points_time[...] = moment

        pos_enc = self.pos_encoding(points.view(-1, self.cfg.n_input_dims - 1))
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
    ) -> "DynamicFlowMLP":
        if isinstance(other, DynamicFlowMLP):
            instance = DynamicFlowMLP(cfg, **kwargs)
            instance.pos_encoding.load_state_dict(other.pos_encoding.state_dict())
            instance.time_encoding.load_state_dict(other.time_encoding.state_dict())
            instance.flow_network.load_state_dict(other.flow_network.state_dict())
            return instance
        else:
            raise TypeError(
                f"Cannot create {DynamicFlowMLP.__name__} from {other.__class__.__name__}"
            )
