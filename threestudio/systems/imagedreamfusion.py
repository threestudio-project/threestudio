import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("image-condition-dreamfusion-system")
class ImageConditionDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
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

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="fit_start",
            step=self.true_global_step,
        )

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()
        # opt.zero_grad()

        do_ref = (
            self.true_global_step < self.cfg.freq.ref_only_steps
            or self.true_global_step % self.cfg.freq.n_ref == 0
        )
        loss = 0.0

        if not do_ref:
            batch = batch["random_camera"]
            if random.random() > 0.5:
                bg_color = None
            else:
                bg_color = torch.rand(3).to(self.device)
            ambient_ratio = 0.1 + 0.9 * random.random()
        else:
            # bg_color = torch.rand_like(batch['rays_o'])
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
            bg_color = None

        batch["bg_color"] = bg_color
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)

        if do_ref:
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]
            gt_depth = batch["depth"]

            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            loss += self.C(self.cfg.loss.lambda_rgb) * F.mse_loss(
                gt_rgb, out["comp_rgb"]
            )

            # mask loss
            loss += self.C(self.cfg.loss.lambda_mask) * F.mse_loss(
                gt_mask.float(), out["opacity"]
            )

            # opacity_clamped = out['opacity'].clamp(1e-3, 1-1e-3)
            # gt_mask_clamped = gt_mask.float().clamp(1e-3, 1-1e-3)
            # loss += self.C(self.cfg.loss.lambda_mask) * binary_cross_entropy(gt_mask_clamped, opacity_clamped)

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = gt_depth[gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                loss += self.C(self.cfg.loss.lambda_depth) * F.mse_loss(
                    valid_gt_depth, valid_pred_depth
                )
        else:
            prompt_utils = self.prompt_processor()
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils, **batch, rgb_as_latents=False
            )

        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

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

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            normal = out["normal"]
            loss_normal_smooth = (
                normal[:, 1:, :, :] - normal[:, :-1, :, :]
            ).square().mean() + (
                normal[:, :, 1:, :] - normal[:, :, :-1, :]
            ).square().mean()
            self.log("train/loss_normal_smooth", loss_normal_smooth)
            loss += self.C(self.cfg.loss.lambda_normal_smooth) * loss_normal_smooth

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
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
            + [{"type": "grayscale", "img": out["depth"][0], "kwargs": {}}]
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
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
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
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
            + [{"type": "grayscale", "img": out["depth"][0], "kwargs": {}}]
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
