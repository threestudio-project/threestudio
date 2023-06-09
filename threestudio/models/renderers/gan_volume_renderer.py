from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.models.renderers.nerf_volume_renderer import NeRFVolumeRenderer
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import get_device
from threestudio.utils.ops import chunk_batch
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("gan-volume-renderer")
class GANVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False
    
    cfg: Config
    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        self.base_renderer = NeRFVolumeRenderer(self.cfg, geometry, material, background)
        # self.generator = Generator()

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        out = self.base_renderer(rays_o, rays_d, light_positions, bg_color, **kwargs)
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.base_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        return self.base_renderer.train(mode)

    def eval(self):
        return self.base_renderer.eval()
