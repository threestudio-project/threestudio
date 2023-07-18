from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.GAN.discriminator import NLayerDiscriminator, weights_init
from threestudio.utils.GAN.distribution import DiagonalGaussianDistribution
from threestudio.utils.GAN.mobilenet import MobileNetV3 as GlobalEncoder
from threestudio.utils.GAN.vae import Decoder as Generator
from threestudio.utils.GAN.vae import Encoder as LocalEncoder
from threestudio.utils.typing import *


@threestudio.register("gan-volume-renderer")
class GANVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        base_renderer_type: str = ""
        base_renderer: Optional[VolumeRenderer.Config] = None

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        self.base_renderer = threestudio.find(self.cfg.base_renderer_type)(
            self.cfg.base_renderer,
            geometry=geometry,
            material=material,
            background=background,
        )
        self.ch_mult = [1, 2, 4]
        self.generator = Generator(
            ch=64,
            out_ch=3,
            ch_mult=self.ch_mult,
            num_res_blocks=1,
            attn_resolutions=[],
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=7,
            resolution=512,
            z_channels=4,
        )
        self.local_encoder = LocalEncoder(
            ch=32,
            out_ch=3,
            ch_mult=self.ch_mult,
            num_res_blocks=1,
            attn_resolutions=[],
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=3,
            resolution=512,
            z_channels=4,
        )
        self.global_encoder = GlobalEncoder(n_class=64)
        self.discriminator = NLayerDiscriminator(
            input_nc=3, n_layers=3, use_actnorm=False, ndf=64
        ).apply(weights_init)

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        gt_rgb: Float[Tensor, "B H W 3"] = None,
        multi_level_guidance: Bool = False,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        B, H, W, _ = rays_o.shape
        if gt_rgb is not None and multi_level_guidance:
            generator_level = torch.randint(0, 3, (1,)).item()
            interval_x = torch.randint(0, 8, (1,)).item()
            interval_y = torch.randint(0, 8, (1,)).item()
            int_rays_o = rays_o[:, interval_y::8, interval_x::8]
            int_rays_d = rays_d[:, interval_y::8, interval_x::8]
            out = self.base_renderer(
                int_rays_o, int_rays_d, light_positions, bg_color, **kwargs
            )
            comp_int_rgb = out["comp_rgb"][..., :3]
            comp_gt_rgb = gt_rgb[:, interval_y::8, interval_x::8]
        else:
            generator_level = 0
        scale_ratio = 2 ** (len(self.ch_mult) - 1)
        rays_o = torch.nn.functional.interpolate(
            rays_o.permute(0, 3, 1, 2),
            (H // scale_ratio, W // scale_ratio),
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        rays_d = torch.nn.functional.interpolate(
            rays_d.permute(0, 3, 1, 2),
            (H // scale_ratio, W // scale_ratio),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

        out = self.base_renderer(rays_o, rays_d, light_positions, bg_color, **kwargs)
        comp_rgb = out["comp_rgb"][..., :3]
        latent = out["comp_rgb"][..., 3:]
        out["comp_lr_rgb"] = comp_rgb.clone()

        posterior = DiagonalGaussianDistribution(latent.permute(0, 3, 1, 2))
        if multi_level_guidance:
            z_map = posterior.sample()
        else:
            z_map = posterior.mode()
        lr_rgb = comp_rgb.permute(0, 3, 1, 2)

        if generator_level == 0:
            g_code_rgb = self.global_encoder(F.interpolate(lr_rgb, (224, 224)))
            comp_gan_rgb = self.generator(torch.cat([lr_rgb, z_map], dim=1), g_code_rgb)
        elif generator_level == 1:
            g_code_rgb = self.global_encoder(
                F.interpolate(gt_rgb.permute(0, 3, 1, 2), (224, 224))
            )
            comp_gan_rgb = self.generator(torch.cat([lr_rgb, z_map], dim=1), g_code_rgb)
        elif generator_level == 2:
            g_code_rgb = self.global_encoder(
                F.interpolate(gt_rgb.permute(0, 3, 1, 2), (224, 224))
            )
            l_code_rgb = self.local_encoder(gt_rgb.permute(0, 3, 1, 2))
            posterior = DiagonalGaussianDistribution(l_code_rgb)
            z_map = posterior.sample()
            comp_gan_rgb = self.generator(torch.cat([lr_rgb, z_map], dim=1), g_code_rgb)

        comp_rgb = F.interpolate(comp_rgb.permute(0, 3, 1, 2), (H, W), mode="bilinear")
        comp_gan_rgb = F.interpolate(comp_gan_rgb, (H, W), mode="bilinear")
        out.update(
            {
                "posterior": posterior,
                "comp_gan_rgb": comp_gan_rgb.permute(0, 2, 3, 1),
                "comp_rgb": comp_rgb.permute(0, 2, 3, 1),
                "generator_level": generator_level,
            }
        )

        if gt_rgb is not None and multi_level_guidance:
            out.update({"comp_int_rgb": comp_int_rgb, "comp_gt_rgb": comp_gt_rgb})
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.base_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        return self.base_renderer.train(mode)

    def eval(self):
        return self.base_renderer.eval()
