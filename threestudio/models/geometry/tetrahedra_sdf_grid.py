from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.geometry.implicit_sdf import ImplicitSDF
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.isosurface import MarchingTetrahedraHelper
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *


@threestudio.register("tetrahedra-sdf-grid")
class TetrahedraSDFGrid(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True
        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        n_input_dims: int = 3
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
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        force_shape_init: bool = False
        geometry_only: bool = False
        fix_geometry: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())

        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
        )

        self.sdf: Float[Tensor, "Nv 1"]
        self.deformation: Optional[Float[Tensor, "Nv 3"]]

        if not self.cfg.fix_geometry:
            self.register_parameter(
                "sdf",
                nn.Parameter(
                    torch.zeros(
                        (self.isosurface_helper.grid_vertices.shape[0], 1),
                        dtype=torch.float32,
                    )
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_parameter(
                    "deformation",
                    nn.Parameter(
                        torch.zeros_like(self.isosurface_helper.grid_vertices)
                    ),
                )
            else:
                self.deformation = None
        else:
            self.register_buffer(
                "sdf",
                torch.zeros(
                    (self.isosurface_helper.grid_vertices.shape[0], 1),
                    dtype=torch.float32,
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_buffer(
                    "deformation",
                    torch.zeros_like(self.isosurface_helper.grid_vertices),
                )
            else:
                self.deformation = None

        if not self.cfg.geometry_only:
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        self.mesh: Optional[Mesh] = None

    def initialize_shape(self) -> None:
        raise NotImplementedError

    def isosurface(self) -> Mesh:
        # return cached mesh if fix_geometry is True to save computation
        if self.cfg.fix_geometry and self.mesh is not None:
            return self.mesh
        mesh = self.isosurface_helper(self.sdf, self.deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox
        )
        if self.cfg.isosurface_remove_outliers:
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
        self.mesh = mesh
        return mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "TetrahedraSDFGrid":
        if isinstance(other, TetrahedraSDFGrid):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            assert instance.cfg.isosurface_resolution == other.cfg.isosurface_resolution
            instance.isosurface_bbox = other.isosurface_bbox.clone()
            instance.sdf.data = other.sdf.data.clone()
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert (
                    instance.deformation is not None and other.deformation is not None
                )
                instance.deformation.data = other.deformation.data.clone()
            if (
                not instance.cfg.geometry_only
                and not other.cfg.geometry_only
                and copy_net
            ):
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        elif isinstance(other, ImplicitVolume):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )
            mesh = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = (
                mesh.extras["grid_level"].to(instance.sdf.data).clamp(-1, 1)
            )
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        elif isinstance(other, ImplicitSDF):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )
            mesh = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data)
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert instance.deformation is not None
                instance.deformation.data = mesh.extras["grid_deformation"].to(
                    instance.deformation.data
                )
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        else:
            raise TypeError(
                f"Cannot create {TetrahedraSDFGrid.__name__} from {other.__class__.__name__}"
            )

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.geometry_only or self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out
