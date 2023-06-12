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
from threestudio.utils.GAN.mobilenet import MobileNetV3 as GlobalEncoder
from threestudio.utils.GAN.vae import Decoder as Generator
from threestudio.utils.GAN.vae import Encoder as LocalEncoder
from threestudio.utils.GAN.distribution import DiagonalGaussianDistribution
from threestudio.utils.GAN.discriminator import NLayerDiscriminator, weights_init


@threestudio.register("patch-renderer")
class PatchRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        patch_size: int = 128
        base_renderer_type: str = ""
    
    cfg: Config
    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        cfg_copy = self.cfg.copy()
        del cfg_copy.patch_size
        del cfg_copy.base_renderer_type
        self.base_renderer = threestudio.find(self.cfg.base_renderer_type)(cfg_copy, 
            geometry=geometry,
            material=material,
            background=background
        )

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        train: Bool = False,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        B, H, W, _ = rays_o.shape
        
        if train:
            scale_ratio = 4
            global_rays_o = torch.nn.functional.interpolate(
                rays_o.permute(0, 3, 1, 2), (H // scale_ratio, W // scale_ratio), mode='bilinear'
            ).permute(0, 2, 3, 1)
            global_rays_d = torch.nn.functional.interpolate(
                rays_d.permute(0, 3, 1, 2), (H // scale_ratio, W // scale_ratio), mode='bilinear'
            ).permute(0, 2, 3, 1)

            out = self.base_renderer(global_rays_o, global_rays_d, light_positions, bg_color, **kwargs)
            comp_rgb = out["comp_rgb"]
            comp_rgb = F.interpolate(comp_rgb.permute(0, 3, 1, 2), (H, W), mode='bilinear').permute(0, 2, 3, 1)

            patch_x = torch.randint(0, W-self.cfg.patch_size, (1,)).item()
            patch_y = torch.randint(0, H-self.cfg.patch_size, (1,)).item()
            patch_rays_o = rays_o[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size]
            patch_rays_d = rays_d[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size]

            out = self.base_renderer(patch_rays_o, patch_rays_d, light_positions, bg_color, **kwargs)
            patch_comp_rgb = out["comp_rgb"]
            comp_rgb[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size] = patch_comp_rgb
    
            out.update({
                "comp_rgb": comp_rgb
            })
        else:
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
