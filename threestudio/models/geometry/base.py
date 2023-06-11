from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.isosurface import (
    IsosurfaceHelper,
    MarchingCubeCPUHelper,
    MarchingTetrahedraHelper,
)
from threestudio.models.mesh import Mesh
from threestudio.utils.base import BaseModule
from threestudio.utils.ops import chunk_batch, scale_tensor
from threestudio.utils.typing import *


def contract_to_unisphere(
    x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], unbounded: bool = False
) -> Float[Tensor, "... 3"]:
    if unbounded:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (0, 1))
    return x


class BaseGeometry(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    @staticmethod
    def create_from(
        other: "BaseGeometry", cfg: Optional[Union[dict, DictConfig]] = None, **kwargs
    ) -> "BaseGeometry":
        raise TypeError(
            f"Cannot create {BaseGeometry.__name__} from {other.__class__.__name__}"
        )

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}


class BaseImplicitGeometry(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        radius: float = 1.0
        isosurface: bool = True
        isosurface_method: str = "mt"
        isosurface_resolution: int = 128
        isosurface_threshold: Union[float, str] = 0.0
        isosurface_chunk: int = 0
        isosurface_coarse_to_fine: bool = True
        isosurface_deformable_grid: bool = False
        isosurface_remove_outliers: bool = True
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

    cfg: Config

    def configure(self) -> None:
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.isosurface_helper: Optional[IsosurfaceHelper] = None
        self.unbounded: bool = False

    def _initilize_isosurface_helper(self):
        if self.cfg.isosurface and self.isosurface_helper is None:
            if self.cfg.isosurface_method == "mc-cpu":
                self.isosurface_helper = MarchingCubeCPUHelper(
                    self.cfg.isosurface_resolution
                ).to(self.device)
            elif self.cfg.isosurface_method == "mt":
                self.isosurface_helper = MarchingTetrahedraHelper(
                    self.cfg.isosurface_resolution,
                    f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
                ).to(self.device)
            else:
                raise AttributeError(
                    "Unknown isosurface method {self.cfg.isosurface_method}"
                )

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        # return the value of the implicit field, could be density / signed distance
        # also return a deformation field if the grid vertices can be optimized
        raise NotImplementedError

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # return the value of the implicit field, where the zero level set represents the surface
        raise NotImplementedError

    def _isosurface(self, bbox: Float[Tensor, "2 3"], fine_stage: bool = False) -> Mesh:
        def batch_func(x):
            # scale to bbox as the input vertices are in [0, 1]
            field, deformation = self.forward_field(
                scale_tensor(
                    x.to(bbox.device), self.isosurface_helper.points_range, bbox
                ),
            )
            field = field.to(
                x.device
            )  # move to the same device as the input (could be CPU)
            if deformation is not None:
                deformation = deformation.to(x.device)
            return field, deformation

        assert self.isosurface_helper is not None

        field, deformation = chunk_batch(
            batch_func,
            self.cfg.isosurface_chunk,
            self.isosurface_helper.grid_vertices,
        )

        threshold: float

        if isinstance(self.cfg.isosurface_threshold, float):
            threshold = self.cfg.isosurface_threshold
        elif self.cfg.isosurface_threshold == "auto":
            eps = 1.0e-5
            threshold = field[field > eps].mean().item()
            threestudio.info(
                f"Automatically determined isosurface threshold: {threshold}"
            )
        else:
            raise TypeError(
                f"Unknown isosurface_threshold {self.cfg.isosurface_threshold}"
            )

        level = self.forward_level(field, threshold)
        mesh: Mesh = self.isosurface_helper(level, deformation=deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, bbox
        )  # scale to bbox as the grid vertices are in [0, 1]
        mesh.add_extra("bbox", bbox)

        if self.cfg.isosurface_remove_outliers:
            # remove outliers components with small number of faces
            # only enabled when the mesh is not differentiable
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)

        return mesh

    def isosurface(self) -> Mesh:
        if not self.cfg.isosurface:
            raise NotImplementedError(
                "Isosurface is not enabled in the current configuration"
            )
        self._initilize_isosurface_helper()
        if self.cfg.isosurface_coarse_to_fine:
            threestudio.debug("First run isosurface to get a tight bounding box ...")
            with torch.no_grad():
                mesh_coarse = self._isosurface(self.bbox)
            vmin, vmax = mesh_coarse.v_pos.amin(dim=0), mesh_coarse.v_pos.amax(dim=0)
            vmin_ = (vmin - (vmax - vmin) * 0.1).max(self.bbox[0])
            vmax_ = (vmax + (vmax - vmin) * 0.1).min(self.bbox[1])
            threestudio.debug("Run isosurface again with the tight bounding box ...")
            mesh = self._isosurface(torch.stack([vmin_, vmax_], dim=0), fine_stage=True)
        else:
            mesh = self._isosurface(self.bbox)
        return mesh


class BaseExplicitGeometry(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        radius: float = 1.0

    cfg: Config

    def configure(self) -> None:
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )
