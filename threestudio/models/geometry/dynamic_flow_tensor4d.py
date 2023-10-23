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


@threestudio.register("dynamic-flow-tensor4d")
class DynamicFlowTensor4D(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 4
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 13,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 13,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            }
        )
        need_normalization: bool = False
        bbox_min: Optional[Tuple[float, float, float]] = None
        bbox_max: Optional[Tuple[float, float, float]] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.enc0 = get_encoding(2, self.cfg.pos_encoding_config)
        self.enc1 = get_encoding(2, self.cfg.pos_encoding_config)
        self.enc2 = get_encoding(2, self.cfg.pos_encoding_config)
        self.enc3 = get_encoding(2, self.cfg.pos_encoding_config)
        self.enc4 = get_encoding(2, self.cfg.pos_encoding_config)
        self.enc5 = get_encoding(2, self.cfg.pos_encoding_config)

        self.pos_encoding_list = [
            self.enc0,
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
            self.enc5,
        ]

        self.flow_network = get_mlp(
            self.enc0.n_output_dims * 6,
            self.cfg.n_feature_dims,
            self.cfg.mlp_network_config,
        )

    def forward(
        self, points_3d: Float[Tensor, "*N Di"], moment
    ) -> Dict[str, Float[Tensor, "..."]]:
        # points 4D
        if self.cfg.need_normalization:
            bbox_min = self.cfg.bbox_min
            bbox_max = self.cfg.bbox_max
            points = points_3d.clone()
            scale_x = bbox_max[0] - bbox_min[0]
            scale_y = bbox_max[1] - bbox_min[1]
            scale_z = bbox_max[2] - bbox_min[2]
            points[:, 0] = (points[:, 0] - bbox_min[0]) / scale_x
            points[:, 1] = (points[:, 1] - bbox_min[1]) / scale_y
            points[:, 2] = (points[:, 2] - bbox_min[2]) / scale_z
        else:
            points = points_3d[..., : self.cfg.n_input_dims - 1]
        points = points.view(-1, self.cfg.n_input_dims - 1)

        points_xy = torch.zeros_like(points[:, :2])
        points_xy[:, 0] = points[:, 0]
        points_xy[:, 1] = points[:, 1]

        points_xz = torch.zeros_like(points[:, :2])
        points_xz[:, 0] = points[:, 0]
        points_xz[:, 1] = points[:, 2]

        points_xt = torch.zeros_like(points[:, :2])
        points_xt[:, 0] = points[:, 0]
        points_xt[:, 1] = moment

        points_yz = torch.zeros_like(points[:, :2])
        points_yz[:, 0] = points[:, 1]
        points_yz[:, 1] = points[:, 2]

        points_yt = torch.zeros_like(points[:, :2])
        points_yt[:, 0] = points[:, 1]
        points_yt[:, 1] = moment

        points_zt = torch.zeros_like(points[:, :2])
        points_zt[:, 0] = points[:, 2]
        points_zt[:, 1] = moment

        points_list = [points_xy, points_xz, points_xt, points_yz, points_yt, points_zt]

        pos_enc_list = []
        for i in range(6):
            pos_enc = self.pos_encoding_list[i](points_list[i])
            pos_enc_list.append(pos_enc)
        enc = torch.cat(pos_enc_list, dim=-1)
        features = self.flow_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        if self.cfg.need_normalization:
            features[0] *= scale_x
            features[1] *= scale_y
            features[2] *= scale_z
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
    ) -> "DynamicFlowTensor4D":
        if isinstance(other, DynamicFlowTensor4D):
            instance = DynamicFlowTensor4D(cfg, **kwargs)
            instance.pos_encoding.load_state_dict(other.pos_encoding.state_dict())
            instance.time_encoding.load_state_dict(other.time_encoding.state_dict())
            instance.flow_network.load_state_dict(other.flow_network.state_dict())
            return instance
        else:
            raise TypeError(
                f"Cannot create {DynamicFlowTensor4D.__name__} from {other.__class__.__name__}"
            )
