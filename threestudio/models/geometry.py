from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.mesh import Mesh
from threestudio.models.isosurface import IsosurfaceHelper, MarchingCubeCPUHelper, MarchingTetrahedraHelper
from threestudio.utils.base import BaseModule
from threestudio.models.networks import create_network_with_input_encoding, get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor, get_activation, chunk_batch
from threestudio.utils.typing import *


def contract_to_unisphere(x: Float[Tensor, "... 3"], bbox: Float[Tensor, "2 3"], unbounded: bool=False) -> Float[Tensor, "... 3"]:
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


class BaseImplicitGeometry(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0
        isosurface: bool = True
        isosurface_method: str = "mt"
        isosurface_resolution: int = 64
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
        self.unbounded: bool = False

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        raise NotImplementedError

    def _isosurface(self, bbox: Float[Tensor, "2 3"]) -> Mesh:
        def batch_func(x):
            # scale to bbox as the input vertices are in [0, 1]
            rv = self.forward_level(scale_tensor(x.to(bbox.device), (0, 1), bbox)).to(x.device) # move to the same device as the input (could be CPU)
            return rv

        assert self.isosurface_helper is not None
        if self.cfg.isosurface_chunk > 0:
            level = chunk_batch(
                batch_func, self.cfg.isosurface_chunk, self.isosurface_helper.grid_vertices
            )
        else:
            level = batch_func(self.isosurface_helper.grid_vertices)
        mesh = self.isosurface_helper(level)
        mesh.v_pos = scale_tensor(mesh.v_pos, (0, 1), bbox) # scale to bbox as the grid vertices are in [0, 1]
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
        density_bias: Union[float, str] = 'blob_magic3d'
        density_blob_scale: float = 10.
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
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 64,
            "n_hidden_layers": 1
        })
        normal_type: Optional[str] = "finite_difference" # in ['pred', 'finite_difference']
        finite_difference_normal_eps: float = 0.01

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.n_output_dims = 1 + self.cfg.n_feature_dims
        self.encoding = get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
        self.network = get_mlp(self.encoding.n_output_dims, self.n_output_dims, self.cfg.mlp_network_config)
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(self.encoding.n_output_dims, 3, self.cfg.mlp_network_config)
    
    def get_activated_density(self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]):
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == 'blob_dreamfusion':
            # post-activation density bias
            density_bias = self.cfg.density_blob_scale * torch.exp(-0.5 * (points ** 2).sum(dim=-1) / self.cfg.density_blob_std ** 2)[...,None]
            density = get_activation(self.cfg.density_activation)(density) + density_bias
        elif self.cfg.density_bias == 'blob_magic3d':
            # pre-activation density bias
            density_bias = self.cfg.density_blob_scale * (1 - torch.sqrt((points.detach() ** 2).sum(dim=-1)) / self.cfg.density_blob_std)[...,None]
            density = get_activation(self.cfg.density_activation)(density + density_bias)
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
            density = get_activation(self.cfg.density_activation)(density + density_bias)
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        return density

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points # points in the original scale
        points = contract_to_unisphere(points, self.bbox, self.unbounded) # points normalized to (0, 1)
        
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        out = self.network(enc).view(*points.shape[:-1], self.n_output_dims)
        density, features = out[..., 0:1], out[..., 1:]
        density = self.get_activated_density(points_unscaled, density)

        output = {
            'density': density,
            'features': features,
        }

        if output_normal:
            if self.cfg.normal_type == "finite_difference":
                eps = self.cfg.finite_difference_normal_eps
                offsets: Float[Tensor, "6 3"] = torch.as_tensor([[eps, 0., 0.], [-eps, 0., 0.], [0., eps, 0.], [0., -eps, 0.], [0., 0., eps], [0., 0., -eps]]).to(points_unscaled)
                points_offset: Float[Tensor, "... 6 3"] = (points_unscaled[...,None,:] + offsets).clamp(-self.cfg.radius, self.cfg.radius)
                density_offset: Float[Tensor, "... 6 1"] = self.forward_density(points_offset)
                normal = -0.5 * (density_offset[...,0::2,0] - density_offset[...,1::2,0]) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(density, points_unscaled, grad_outputs=torch.ones_like(density), create_graph=True)[0]
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({
                'normal': normal,
                'shading_normal': normal
            })
        
        torch.set_grad_enabled(grad_enabled)
        return output
    
    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
    
        density = self.network(self.encoding(
            points.reshape(-1, self.cfg.n_input_dims)
        )).reshape(*points.shape[:-1], self.n_output_dims)[..., 0:1]

        density = self.get_activated_density(points_unscaled, density)
        return density

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        density = self.forward_density(points)
        return -(density - self.cfg.isosurface_threshold)


@threestudio.register("implicit-sdf")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(default_factory=lambda: {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.447269237440378,
        })
        mlp_network_config: dict = field(default_factory=lambda: {
            "otype": "VanillaMLP",
            "activation": "ReLU",
            "output_activation": "none",
            "n_neurons": 32,
            "n_hidden_layers": 2,
        })
        normal_type: Optional[str] = "finite_difference" # in ['pred', 'finite_difference']
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        force_shape_init: bool = False   

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.n_output_dims = 1 + self.cfg.n_feature_dims
        self.encoding = get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
        self.network = get_mlp(self.encoding.n_output_dims, self.n_output_dims, self.cfg.mlp_network_config)
        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(self.encoding.n_output_dims, 3, self.cfg.mlp_network_config)

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return
        
        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return
        
        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm
        for _ in tqdm(range(1000), desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:"):
            points_rand = torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2. - 1.
            if self.cfg.shape_init == "ellipsoid":
                assert isinstance(self.cfg.shape_init_params, Sized) and len(self.cfg.shape_init_params) == 3
                size = torch.as_tensor(self.cfg.shape_init_params).to(points_rand)
                sdf_gt = ((points_rand / size)**2).sum(dim=-1, keepdim=True).sqrt() - 1.0 # pseudo signed distance of an ellipsoid
            elif self.cfg.shape_init == "sphere":
                assert isinstance(self.cfg.shape_init_params, float)
                radius = self.cfg.shape_init_params
                sdf_gt = ((points_rand ** 2).sum(dim=-1, keepdim=True).sqrt() - radius)
            elif self.cfg.shape_init == "mesh":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown shape initialization type: {self.cfg.shape_init}")
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()            

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        points_unscaled = points # points in the original scale
        points = contract_to_unisphere(points, self.bbox, self.unbounded) # points normalized to (0, 1)
        
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        out = self.network(enc).view(*points.shape[:-1], self.n_output_dims)
        sdf, features = out[..., 0:1], out[..., 1:]

        output = {
            'sdf': sdf,
            'features': features,
        }

        if output_normal:
            if self.cfg.normal_type == "finite_difference":
                eps = 1.e-3
                offsets: Float[Tensor, "6 3"] = torch.as_tensor([[eps, 0., 0.], [-eps, 0., 0.], [0., eps, 0.], [0., -eps, 0.], [0., 0., eps], [0., 0., -eps]]).to(points_unscaled)
                points_offset: Float[Tensor, "... 6 3"] = (points_unscaled[...,None,:] + offsets).clamp(-self.cfg.radius, self.cfg.radius)
                sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(points_offset)
                normal = 0.5 * (sdf_offset[...,0::2,0] - sdf_offset[...,1::2,0]) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({
                'normal': normal,
                'shading_normal': normal
            })
        return output
    
    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
    
        sdf = self.network(self.encoding(
            points.reshape(-1, self.cfg.n_input_dims)
        )).reshape(*points.shape[:-1], self.n_output_dims)[..., 0:1]

        return sdf

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        sdf = self.forward_sdf(points)
        return sdf - self.cfg.isosurface_threshold


@threestudio.register("volume-grid")
class VolumeGrid(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        grid_size: Tuple[int,int,int] = field(default_factory=lambda: (100, 100, 100))
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = 'blob'
        density_blob_scale: float = 5.
        density_blob_std: float = 0.5
        normal_type: Optional[str] = "finite_difference" # in ['pred', 'finite_difference']

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.grid_size = self.cfg.grid_size

        self.grid = nn.Parameter(torch.zeros(1, self.cfg.n_feature_dims + 1, *self.grid_size))
        if self.cfg.density_bias == 'blob':
            self.register_buffer('density_scale', torch.tensor(0.0))
        else:
            self.density_scale = nn.Parameter(torch.tensor(0.0))

        if self.cfg.normal_type == "pred":
            self.normal_grid = nn.Parameter(torch.zeros(1, 3, *self.grid_size))

    def get_density_bias(self, points: Float[Tensor, "*N Di"]):
        if self.cfg.density_bias == 'blob':
            # density_bias: Float[Tensor, "*N 1"] = self.cfg.density_blob_scale * torch.exp(-0.5 * (points ** 2).sum(dim=-1) / self.cfg.density_blob_std ** 2)[...,None]
            density_bias: Float[Tensor, "*N 1"] = self.cfg.density_blob_scale * (1 - torch.sqrt((points.detach() ** 2).sum(dim=-1)) / self.cfg.density_blob_std)[...,None]
            return density_bias
        elif isinstance(self.cfg.density_bias, float):
            return self.cfg.density_bias
        else:
            raise AttributeError(f"Unknown density bias {self.cfg.density_bias}")

    def get_trilinear_feature(self, points: Float[Tensor, "*N Di"], grid: Float[Tensor, "1 Df G1 G2 G3"]) -> Float[Tensor, "*N Df"]:
        points_shape = points.shape[:-1]
        df = grid.shape[1]
        di = points.shape[-1]
        out = F.grid_sample(grid, points.view(1, 1, 1, -1, di), align_corners=True, mode='bilinear')
        out = out.reshape(df, -1).T.reshape(*points_shape, df)
        return out

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        points_unscaled = points # points in the original scale
        points = contract_to_unisphere(points, self.bbox, self.unbounded) # points normalized to (0, 1)
        points = points * 2 - 1 # convert to [-1, 1] for grid sample

        out = self.get_trilinear_feature(points, self.grid)
        density, features = out[..., 0:1], out[..., 1:]
        density = density * torch.exp(self.density_scale) # exp scaling in DreamFusion
        
        # breakpoint()
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )

        output = {
            'density': density,
            'features': features,
        }

        if output_normal:
            if self.cfg.normal_type == "finite_difference":
                eps = 1.e-3
                offsets: Float[Tensor, "6 3"] = torch.as_tensor([[eps, 0., 0.], [-eps, 0., 0.], [0., eps, 0.], [0., -eps, 0.], [0., 0., eps], [0., 0., -eps]]).to(points_unscaled)
                points_offset: Float[Tensor, "... 6 3"] = (points_unscaled[...,None,:] + offsets).clamp(-self.cfg.radius, self.cfg.radius)
                density_offset: Float[Tensor, "... 6 1"] = self.forward_density(points_offset)
                normal = -0.5 * (density_offset[...,0::2,0] - density_offset[...,1::2,0]) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.get_trilinear_feature(points, self.normal_grid)
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({
                'normal': normal,
                'shading_normal': normal
            })
        return output
    
    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1 # convert to [-1, 1] for grid sample

        out = self.get_trilinear_feature(points, self.grid)
        density = out[..., 0:1]
        density = density * torch.exp(self.density_scale)
        
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )
        return density

    def forward_level(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        density = self.forward_density(points)
        return -(density - self.cfg.isosurface_threshold)
