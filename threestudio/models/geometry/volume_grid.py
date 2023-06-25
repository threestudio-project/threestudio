from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("volume-grid")
class VolumeGrid(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        grid_size: Tuple[int, int, int] = field(default_factory=lambda: (100, 100, 100))
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob"
        density_blob_scale: float = 5.0
        density_blob_std: float = 0.5
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = "auto"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.grid_size = self.cfg.grid_size

        self.grid = nn.Parameter(
            torch.zeros(1, self.cfg.n_feature_dims + 1, *self.grid_size)
        )
        if self.cfg.density_bias == "blob":
            self.register_buffer("density_scale", torch.tensor(0.0))
        else:
            self.density_scale = nn.Parameter(torch.tensor(0.0))

        if self.cfg.normal_type == "pred":
            self.normal_grid = nn.Parameter(torch.zeros(1, 3, *self.grid_size))

    def get_density_bias(self, points: Float[Tensor, "*N Di"]):
        if self.cfg.density_bias == "blob":
            # density_bias: Float[Tensor, "*N 1"] = self.cfg.density_blob_scale * torch.exp(-0.5 * (points ** 2).sum(dim=-1) / self.cfg.density_blob_std ** 2)[...,None]
            density_bias: Float[Tensor, "*N 1"] = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points.detach() ** 2).sum(dim=-1))
                    / self.cfg.density_blob_std
                )[..., None]
            )
            return density_bias
        elif isinstance(self.cfg.density_bias, float):
            return self.cfg.density_bias
        else:
            raise AttributeError(f"Unknown density bias {self.cfg.density_bias}")

    def get_trilinear_feature(
        self, points: Float[Tensor, "*N Di"], grid: Float[Tensor, "1 Df G1 G2 G3"]
    ) -> Float[Tensor, "*N Df"]:
        points_shape = points.shape[:-1]
        df = grid.shape[1]
        di = points.shape[-1]
        out = F.grid_sample(
            grid, points.view(1, 1, 1, -1, di), align_corners=False, mode="bilinear"
        )
        out = out.reshape(df, -1).T.reshape(*points_shape, df)
        return out

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        out = self.get_trilinear_feature(points, self.grid)
        density, features = out[..., 0:1], out[..., 1:]
        density = density * torch.exp(self.density_scale)  # exp scaling in DreamFusion

        # breakpoint()
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )

        output = {
            "density": density,
            "features": features,
        }

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                eps = 1.0e-3
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.get_trilinear_feature(points, self.normal_grid)
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        out = self.get_trilinear_feature(points, self.grid)
        density = out[..., 0:1]
        density = density * torch.exp(self.density_scale)

        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample
        features = self.get_trilinear_feature(points, self.grid)[..., 1:]
        out.update(
            {
                "features": features,
            }
        )
        return out
