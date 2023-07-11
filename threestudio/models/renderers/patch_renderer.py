from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.typing import *


@threestudio.register("patch-renderer")
class PatchRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        patch_size: int = 128
        base_renderer_type: str = ""
        base_renderer: Optional[VolumeRenderer.Config] = None
        global_detach: bool = False
        global_downsample: int = 4
        global_prune: bool = True
        mode: str = "patch"  # patch or interval
        block_nums: Tuple = (2, 2)

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

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        B, H, W, _ = rays_o.shape

        valid_patch_key = []
        out_global = {}
        if self.base_renderer.training:
            if self.cfg.mode == "interval":
                YI = self.cfg.block_nums[0]
                XI = self.cfg.block_nums[1]

                MAX_N = self.base_renderer.cfg.train_max_nums
                for i in range(YI):
                    for j in range(XI):
                        int_rays_o = rays_o[:, i::YI, j::XI]
                        int_rays_d = rays_d[:, i::YI, j::XI]
                        if MAX_N > 0:
                            self.base_renderer.cfg.train_max_nums = MAX_N // (YI * XI)
                        out = self.base_renderer(
                            int_rays_o, int_rays_d, light_positions, bg_color, **kwargs
                        )
                        if len(valid_patch_key) == 0:
                            for key in out:
                                if torch.is_tensor(out[key]):
                                    if len(out[key].shape) == len(
                                        out["comp_rgb"].shape
                                    ):
                                        if (
                                            out[key][..., 0].shape
                                            == out["comp_rgb"][..., 0].shape
                                        ):
                                            valid_patch_key.append(key)
                                            out_global[key] = torch.zeros(
                                                B,
                                                H,
                                                W,
                                                out[key].shape[-1],
                                                dtype=out[key].dtype,
                                                device=out[key].device,
                                            )
                                    if len(out[key].shape) == len(out["weights"].shape):
                                        if (
                                            out[key][..., 0].shape
                                            == out["weights"][..., 0].shape
                                        ):
                                            valid_patch_key.append(key)
                                            out_global[key] = []
                        for key in valid_patch_key:
                            if torch.is_tensor(out_global[key]):
                                out_global[key][:, i::YI, j::XI] = out[key]
                            else:
                                out_global[key].append(out[key])
                for key in valid_patch_key:
                    if not torch.is_tensor(out_global[key]):
                        out_global[key] = torch.cat(out_global[key], dim=0)
                self.base_renderer.cfg.train_max_nums = MAX_N
                out = out_global
            else:
                downsample = self.cfg.global_downsample
                global_rays_o = torch.nn.functional.interpolate(
                    rays_o.permute(0, 3, 1, 2),
                    (H // downsample, W // downsample),
                    mode="bilinear",
                ).permute(0, 2, 3, 1)
                global_rays_d = torch.nn.functional.interpolate(
                    rays_d.permute(0, 3, 1, 2),
                    (H // downsample, W // downsample),
                    mode="bilinear",
                ).permute(0, 2, 3, 1)
                if not self.cfg.global_prune:
                    grid_prune = self.base_renderer.cfg.grid_prune
                    self.base_renderer.cfg.grid_prune = False
                out_global = self.base_renderer(
                    global_rays_o, global_rays_d, light_positions, bg_color, **kwargs
                )
                if not self.cfg.global_prune:
                    self.base_renderer.cfg.grid_prune = grid_prune

                PS = min(self.cfg.patch_size, min(H, W))
                patch_x = torch.randint(0, W - PS + 1, (1,)).item()
                patch_y = torch.randint(0, H - PS + 1, (1,)).item()
                patch_rays_o = rays_o[:, patch_y : patch_y + PS, patch_x : patch_x + PS]
                patch_rays_d = rays_d[:, patch_y : patch_y + PS, patch_x : patch_x + PS]
                out = self.base_renderer(
                    patch_rays_o, patch_rays_d, light_positions, bg_color, **kwargs
                )

                valid_patch_key = []
                for key in out:
                    if torch.is_tensor(out[key]):
                        if len(out[key].shape) == len(out["comp_rgb"].shape):
                            if out[key][..., 0].shape == out["comp_rgb"][..., 0].shape:
                                valid_patch_key.append(key)
                for key in valid_patch_key:
                    out_global[key] = F.interpolate(
                        out_global[key].permute(0, 3, 1, 2), (H, W), mode="bilinear"
                    ).permute(0, 2, 3, 1)
                    if self.cfg.global_detach:
                        out_global[key] = out_global[key].detach()
                    out_global[key][
                        :, patch_y : patch_y + PS, patch_x : patch_x + PS
                    ] = out[key]
                out = out_global
        else:
            out = self.base_renderer(
                rays_o, rays_d, light_positions, bg_color, **kwargs
            )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        self.base_renderer.update_step(epoch, global_step, on_load_weights)

    def train(self, mode=True):
        return self.base_renderer.train(mode)

    def eval(self):
        return self.base_renderer.eval()
