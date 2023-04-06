from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nerfacc import ContractionType

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.models.isosurface import IsosurfaceHelper, MarchingCubeCPUHelper, MarchingTetrahedraHelper
from threestudio.utils.base import BaseModule
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.utils.ops import scale_tensor, get_activation, chunk_batch
from threestudio.utils.typing import *


def contract_to_unisphere(x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], contraction_type: ContractionType) -> Float[Tensor, "... 3"]:
    if contraction_type == ContractionType.AABB:
        x = scale_tensor(x, bbox, (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise AttributeError(f"Unknown contraction type {contraction_type}")
    return x


class BaseImplicitGeometry(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0
        isosurface: bool = True
        isosurface_method: str = "mt"
        isosurface_resolution: int = 32
        isosurface_threshold: float = 0.0
        isosurface_chunk: int = 2097152
        isosurface_optimize_grid: bool = False
        isosurface_coarse_to_fine: bool = False

    cfg: Config

    def configure(self) -> None:
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer("bbox", torch.as_tensor(
            [
                [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                [self.cfg.radius, self.cfg.radius, self.cfg.radius],
            ],
            dtype=torch.float32,
        ))
        self.isosurface_helper: Optional[IsosurfaceHelper] = None
        if self.cfg.isosurface:
            if self.cfg.isosurface_method == "mc-cpu":
                self.isosurface_helper = MarchingCubeCPUHelper(
                    self.cfg.isosurface_resolution
                )
            elif self.cfg.isosurface_method == 'mt':
                self.isosurface_helper = MarchingTetrahedraHelper(
                    self.cfg.isosurface_resolution,
                    self.cfg.isosurface_optimize_grid,
                    f"load/tets/{self.cfg.isosurface_resolution}_tets.npz"
                )
            else:
                raise AttributeError("Unknown isosurface method {self.cfg.isosurface_method}")
        self.contraction_type: ContractionType = ContractionType.AABB  # FIXME: hard-coded

    def forward(
        self, points: Float[Tensor, "*N Di"]
    ) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError

    def _isosurface(self, bbox: Float[Tensor, "2 3"]) -> Mesh:
        def batch_func(x):
            x = scale_tensor(x, (0, 1), bbox)
            rv = self.forward_level(x)
            return rv

        assert self.isosurface_helper is not None
        level = chunk_batch(
            batch_func, self.cfg.isosurface_chunk, self.isosurface_helper.grid_vertices
        )
        mesh = self.isosurface_helper(level, threshold=self.cfg.isosurface_threshold)
        mesh.v_pos = scale_tensor(mesh.v_pos, (0, 1), bbox)
        return mesh

    def isosurface(self) -> Mesh:
        if not self.cfg.isosurface:
            raise NotImplementedError(
                "Isosurface is disabled in the current configuration"
            )
        if self.cfg.isosurface_coarse_to_fine:
            with torch.no_grad():
                mesh_coarse = self._isosurface(self.bbox)
            vmin, vmax = mesh_coarse.v_pos.amin(dim=0), mesh_coarse.v_pos.amax(
                dim=0
            )
            vmin_ = (vmin - (vmax - vmin) * 0.1).max(self.bbox[0])
            vmax_ = (vmax + (vmax - vmin) * 0.1).min(self.bbox[1])
            mesh = self._isosurface(torch.stack([vmin_, vmax_], dim=0))
        else:
            mesh = self._isosurface(self.bbox)
        return mesh


@threestudio.register("implicit-volume")
class ImplicitVolume(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = 'blob'
        # density_blob_scale: float = 10.
        # density_blob_std: float = 0.5
        density_blob_scale: float = 5.
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(default_factory=lambda: {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.447269237440378
        })
        mlp_network_config: dict = field(default_factory=lambda: {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 64,
            "n_hidden_layers": 1
        })

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.n_output_dims = 1 + 3 + self.cfg.n_feature_dims
        self.encoding_with_network = create_network_with_input_encoding(
            self.cfg.n_input_dims,
            self.n_output_dims,
            self.cfg.pos_encoding_config,
            self.cfg.mlp_network_config,
        )
    
    def get_density_bias(self, points: Float[Tensor, "*N Di"]):
        if self.cfg.density_bias == 'blob':
            # density_bias: Float[Tensor, "*N 1"] = self.cfg.density_blob_scale * torch.exp(-0.5 * (points ** 2).sum(dim=-1) / self.cfg.density_blob_std ** 2)[...,None]
            density_bias: Float[Tensor, "*N 1"] = self.cfg.density_blob_scale * (1 - torch.sqrt((points ** 2).sum(dim=-1)) / self.cfg.density_blob_std)[...,None]
            return density_bias
        elif isinstance(self.cfg.density_bias, float):
            return self.cfg.density_bias
        else:
            raise AttributeError(f"Unknown density bias {self.cfg.density_bias}")

    def forward(
        self, points: Float[Tensor, "*N Di"]
    ) -> Dict[str, Float[Tensor, "..."]]:
        points_unscaled = points # points in the original scale
        points = contract_to_unisphere(points, self.bbox, self.contraction_type) # points normalized to (0, 1)
        out = (
            self.encoding_with_network(points.view(-1, self.cfg.n_input_dims))
            .view(*points.shape[:-1], self.n_output_dims)
            .float()
        )
        density, pred_normal, features = out[..., 0:1], out[..., 1:4], out[..., 4:]
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )
        pred_normal = F.normalize(pred_normal, dim=-1)

        return {
            'density': density,
            'features': features,
            'normal': pred_normal,
            'pred_normal': pred_normal,
            'shading_normal': pred_normal # use MLP predicted normal for shading
        }
    
        # with torch.inference_mode(False): # disable inference mode to enable grad in validation / testing
        #     with torch.enable_grad():
        #         if not self.training:
        #             points = points.clone() # points may be in inference mode, get a copy to enable grad
        #         points.requires_grad_(True)

        #         points_unscaled = points # points in the original scale
        #         points = contract_to_unisphere(points, self.bbox, self.contraction_type) # points normalized to (0, 1)
                
        #         out = (
        #             self.encoding_with_network(points.view(-1, self.cfg.n_input_dims))
        #             .view(*points.shape[:-1], self.n_output_dims)
        #             .float()
        #         )
        #         density, features, pred_normal = out[..., 0:1], out[..., 1:4], out[..., 4:7]
        #         density = get_activation(self.cfg.density_activation)(
        #             density + self.get_density_bias(points_unscaled)
        #         )
        #         pred_normal = F.normalize(pred_normal, dim=-1)
             
        #         normal = -torch.autograd.grad(
        #             density, points_unscaled, grad_outputs=torch.ones_like(density),
        #             create_graph=True, retain_graph=True, only_inputs=True
        #         )[0]
        #         normal = F.normalize(normal, dim=-1)

        # return {
        #     'density': density,
        #     'features': features,
        #     'normal': normal,
        #     'pred_normal': pred_normal,
        #     'shading_normal': pred_normal # use MLP predicted normal for shading
        # }
    
    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.contraction_type)

        density = self.encoding_with_network(
            points.reshape(-1, self.cfg.n_input_dims)
        ).reshape(*points.shape[:-1], self.n_output_dims)[..., 0:1]

        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )
        return density

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        density = self.forward_density(points)
        return -density
