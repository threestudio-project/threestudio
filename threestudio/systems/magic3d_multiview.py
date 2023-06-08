import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.lpips import LPIPS 


@threestudio.register("magic3d-multiview-system")
class Magic3D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # only used when refinement=True and from_coarse=True
        geometry_coarse_type: str = "implicit-volume"
        geometry_coarse: dict = field(default_factory=dict)

        refinement: bool = False
        # path to the coarse stage weights
        from_coarse: Optional[str] = None
        # used to override configurations of the coarse geometry when initialize from coarse
        # for example isosurface_threshold
        coarse_geometry_override: dict = field(default_factory=dict)
        inherit_coarse_texture: bool = True
        per_editing_step: int = 10
        start_editing_step: int = 1000
        patch_size: int = 128
        low_resolution_step: int = 1000

    cfg: Config

    def configure(self) -> None:
        # override the default configure function
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        if self.cfg.refinement:
            self.background.requires_grad_(False)

        if (
            self.cfg.refinement
            and self.cfg.from_coarse  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            threestudio.info("Initializing from coarse stage ...")
            from threestudio.utils.config import load_config, parse_structured

            coarse_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.from_coarse), "../configs/parsed.yaml"
                )
            )  # TODO: hard-coded relative path
            coarse_system_cfg: Magic3D.Config = parse_structured(
                self.Config, coarse_cfg.system
            )
            coarse_geometry_cfg = coarse_system_cfg.geometry
            coarse_geometry_cfg.update(self.cfg.coarse_geometry_override)
            self.geometry = threestudio.find(coarse_system_cfg.geometry_type)(
                coarse_geometry_cfg
            )

            # load coarse stage geometry
            # also load background parameters if are any
            self.load_weights(self.cfg.from_coarse)

            # convert from coarse stage geometry
            self.geometry = self.geometry.to(get_device())
            geometry_refine = threestudio.find(self.cfg.geometry_type).create_from(
                self.geometry,
                self.cfg.geometry,
                copy_net=self.cfg.inherit_coarse_texture,
            )
            del self.geometry
            cleanup()
            self.geometry = geometry_refine
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

        self.perceptual_loss = LPIPS().eval().to(get_device())
        self.edit_frames = {}
        self.cache_frames = {}
        self.per_editing_step = self.cfg.per_editing_step
        self.start_editing_step = self.cfg.start_editing_step

    def forward(self, batch: Dict[str, Any], train=False) -> Dict[str, Any]:
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        if train:
            rays_o = batch["rays_o"]
            rays_d = batch["rays_d"]
            B, H, W, _ = rays_o.shape
            if self.global_step < self.cfg.low_resolution_step:
                rays_o = torch.nn.functional.interpolate(
                    rays_o.permute(0, 3, 1, 2), (H // 4, W // 4)).permute(0, 2, 3, 1)
                rays_d = torch.nn.functional.interpolate(
                    rays_d.permute(0, 3, 1, 2), (H // 4, W // 4)).permute(0, 2, 3, 1)
                batch["rays_o"] = rays_o
                batch["rays_d"] = rays_d
                render_out = self.renderer(**batch)
                render_out["comp_rgb"] = torch.nn.functional.interpolate(
                    render_out["comp_rgb"].permute(0, 3, 1, 2), (H, W)
                ).permute(0, 2, 3, 1)
            else:
                patch_x = torch.randint(0, W-self.cfg.patch_size, (1,)).item()
                patch_y = torch.randint(0, H-self.cfg.patch_size, (1,)).item()
                batch["rays_o"] = rays_o[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size]
                batch["rays_d"] = rays_d[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size]
                render_out = self.renderer(**batch)
                if batch_index not in self.cache_frames:
                    self.cache_frames[batch_index] = torch.zeros_like(rays_o).cpu()
                cache_rgb = self.cache_frames[batch_index].to(rays_o.device)
                cache_rgb[:, patch_y:patch_y+self.cfg.patch_size, patch_x:patch_x+self.cfg.patch_size] = render_out["comp_rgb"]
                render_out["comp_rgb"] = cache_rgb
                self.cache_frames[batch_index] = cache_rgb.detach().cpu()
        else:
            render_out = self.renderer(**batch)
            self.cache_frames[batch_index] = render_out["comp_rgb"].detach().cpu()
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        out = self(batch, train=True)
        prompt_utils = self.prompt_processor
        if self.per_editing_step > 0 and self.global_step > self.start_editing_step:
            if not batch_index in self.edit_frames or self.global_step % self.per_editing_step == 0:
                result = self.guidance(out["comp_rgb"], batch["gt_rgb"], prompt_utils)
                self.edit_frames[batch_index] = result["edit_images"].detach().cpu()
            rgb = self.edit_frames[batch_index].to(batch["gt_rgb"].device)
            B, H, W, C = batch["gt_rgb"].shape
            rgb = torch.nn.functional.interpolate(
                rgb.permute(0, 3, 1, 2), (H, W), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            rgb = batch["gt_rgb"]
        loss = 0.0
        guidance_out = {
            "loss_l1": torch.nn.functional.l1_loss(out["comp_rgb"], rgb),
            "loss_p": self.perceptual_loss(
                out["comp_rgb"].permute(0, 3, 1, 2).contiguous(), 
                rgb.permute(0, 3, 1, 2).contiguous()
            ).sum(),
        }

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        if batch_index in self.edit_frames:
            B, H, W, C = batch["gt_rgb"].shape
            rgb = torch.nn.functional.interpolate(
                self.edit_frames[batch_index].permute(0, 3, 1, 2), (H, W)
            ).permute(0, 2, 3, 1)[0]
        else:
            rgb = batch["gt_rgb"][0]
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": rgb,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
