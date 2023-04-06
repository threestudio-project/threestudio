from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, accumulate_along_rays

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.models.geometry import BaseImplicitGeometry
from threestudio.models.materials import BaseMaterial
from threestudio.models.background import BaseBackground
from threestudio.utils.typing import *
from threestudio.utils.ops import chunk_batch

class Renderer(BaseModule):
    def configure(self, geometry: BaseImplicitGeometry, material: BaseMaterial, background: BaseBackground) -> None:
        self.geometry = geometry
        self.material = material
        self.background = background
    
    def forward(self, *args, **kwargs) -> Dict[str, Float[Tensor, "..."]]:
        raise NotImplementedError

class VolumeRenderer(Renderer):
    pass

class Rasterizer(Renderer):
    pass


@threestudio.register('nerf-volume-renderer')
class NeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        radius: float = 1.0
        num_samples_per_ray: int = 512
        grid_prune: bool = True
        randomized: bool = True
        eval_chunk_size: int = 8192
    
    cfg: Config

    def configure(self, geometry: BaseImplicitGeometry, material: BaseMaterial, background: BaseBackground) -> None:
        super().configure(geometry, material, background)
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer("bbox", torch.as_tensor(
            [
                [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                [self.cfg.radius, self.cfg.radius, self.cfg.radius],
            ],
            dtype=torch.float32,
        ))
        self.contraction_type = ContractionType.AABB
        self.occupancy_grid_res = 128
        self.near_plane, self.far_plane = None, None
        self.cone_angle = 0.0
        self.render_step_size = 1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        if self.cfg.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.bbox.view(-1),
                resolution=self.occupancy_grid_res,
                contraction_type=self.contraction_type
            )
        self.randomized = self.cfg.randomized

    def forward(self, rays_o: Float[Tensor, "B H W 3"], rays_d: Float[Tensor, "B H W 3"], light_positions: Float[Tensor, "B 3"], **kwargs) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.view(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.view(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = light_positions.view(-1, 1, 1, 3).expand(-1, height, width, -1).view(-1, 3)
        n_rays = rays_o_flatten.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o_flatten[ray_indices]
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density = self.geometry.forward_density(positions)
            return density

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o_flatten, rays_d_flatten,
                scene_aabb=self.bbox.view(-1),
                grid=self.occupancy_grid if self.cfg.grid_prune else None,
                # sigma_fn=sigma_fn,
                near_plane=self.near_plane, far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=self.cone_angle
            )   
        ray_indices = ray_indices.long()
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if self.training:
            geo_out = self.geometry(positions)
            rgb_fg_all = self.material(viewdirs=t_dirs, positions=positions, light_positions=t_light_positions, **geo_out)
            comp_rgb_bg = self.background(dirs=rays_d_flatten)
        else:
            geo_out = chunk_batch(self.geometry, self.cfg.eval_chunk_size, positions)
            rgb_fg_all = chunk_batch(self.material, self.cfg.eval_chunk_size, viewdirs=t_dirs, positions=positions, light_positions=t_light_positions, **geo_out)
            comp_rgb_bg = chunk_batch(self.background, self.cfg.eval_chunk_size, dirs=rays_d_flatten)

        weights = render_weight_from_density(t_starts, t_ends, geo_out['density'], ray_indices=ray_indices, n_rays=n_rays)
        opacity: Float[Tensor, "Nr 1"] = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth: Float[Tensor, "Nr 1"] = accumulate_along_rays(weights, ray_indices, values=t_positions, n_rays=n_rays)
        comp_rgb_fg: Float[Tensor, "Nr 3"] = accumulate_along_rays(weights, ray_indices, values=rgb_fg_all, n_rays=n_rays)

        comp_rgb = comp_rgb_fg + comp_rgb_bg * (1.0 - opacity)    

        out = {
            'comp_rgb': comp_rgb.view(batch_size, height, width, 3),
            'comp_rgb_fg': comp_rgb_fg.view(batch_size, height, width, 3),
            'comp_rgb_bg': comp_rgb_bg.view(batch_size, height, width, 3),
            'opacity': opacity.view(batch_size, height, width, 1),
            'depth': depth.view(batch_size, height, width, 1)
        }

        if self.training:
            out.update({
                'weights': weights,
                't_points': t_positions,
                't_intervals': t_intervals,
                't_dirs': t_dirs,
                'ray_indices': ray_indices,
                **geo_out
            })
        else:
            comp_normal: Float[Tensor, "Nr 3"] = accumulate_along_rays(weights, ray_indices, values=geo_out['normal'], n_rays=n_rays)
            comp_pred_normal: Float[Tensor, "Nr 3"] = accumulate_along_rays(weights, ray_indices, values=geo_out['pred_normal'], n_rays=n_rays)
            comp_normal = F.normalize(comp_normal, dim=-1)
            comp_pred_normal = F.normalize(comp_pred_normal, dim=-1)
            comp_normal = (comp_normal + 1.) / 2. * opacity # for visualization
            comp_pred_normal = (comp_pred_normal + 1.) / 2. * opacity # for visualization
            out.update({
                'comp_normal': comp_normal.view(batch_size, height, width, 3),
                'comp_pred_normal': comp_pred_normal.view(batch_size, height, width, 3)
            })
        
        return out

    
    def update_step(self, epoch: int, global_step: int) -> None:
        def occ_eval_fn(x):
            density = self.geometry.forward_density(x)
            # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
            return density * self.render_step_size
        
        if self.training and self.cfg.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn)
    
    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()


class NeuSVolumeRenderer(VolumeRenderer):
    pass

class DeferredVolumeRenderer(VolumeRenderer):
    pass

class NVDiffRasterizer(Rasterizer):
    pass