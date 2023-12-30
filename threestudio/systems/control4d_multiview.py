import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer
from threestudio.utils.GAN.loss import discriminator_loss, generator_loss
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *


@threestudio.register("control4d-multiview-system")
class Control4D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        per_editing_step: int = 20
        start_editing_step: int = 2000

    cfg: Config

    def configure(self) -> None:
        # override the default configure function
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        p_config = {}
        self.perceptual_loss = threestudio.find("perceptual-loss")(p_config)
        self.edit_frames = {}
        self.per_editing_step = self.cfg.per_editing_step
        self.start_editing_step = self.cfg.start_editing_step

        self.automatic_optimization = False

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

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        self.toggle_optimizer(optimizer_g)

        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        batch["multi_level_guidance"] = True

        origin_gt_rgb = batch["gt_rgb"]
        B, H, W, C = origin_gt_rgb.shape
        if batch_index in self.edit_frames:
            gt_rgb = self.edit_frames[batch_index].to(batch["gt_rgb"].device)
            gt_rgb = torch.nn.functional.interpolate(
                gt_rgb.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            batch["gt_rgb"] = gt_rgb
        else:
            gt_rgb = origin_gt_rgb
        out = self(batch)
        if self.per_editing_step > 0 and self.global_step > self.start_editing_step:
            prompt_utils = self.prompt_processor()
            if (
                not batch_index in self.edit_frames
                or self.global_step % self.per_editing_step == 0
            ):
                result = self.guidance(out["comp_gan_rgb"], origin_gt_rgb, prompt_utils)
                self.edit_frames[batch_index] = result["edit_images"].detach().cpu()

        loss = 0.0
        # loss of generator level 0
        loss_l1 = F.l1_loss(out["comp_int_rgb"], out["comp_gt_rgb"])
        loss_p = 0.0
        loss_kl = out["posterior"].kl().mean()
        loss_G = generator_loss(
            self.renderer.discriminator,
            gt_rgb.permute(0, 3, 1, 2),
            out["comp_gan_rgb"].permute(0, 3, 1, 2),
        )

        generator_level = out["generator_level"]

        level_ratio = 1.0 if generator_level == 2 else 0.1
        loss_l1 += F.l1_loss(out["comp_gan_rgb"], gt_rgb) * level_ratio
        lr_gan_rgb = F.interpolate(
            out["comp_gan_rgb"].permute(0, 3, 1, 2), (H // 4, W // 4), mode="area"
        )
        lr_rgb = F.interpolate(
            out["comp_rgb"].permute(0, 3, 1, 2), (H // 4, W // 4), mode="area"
        ).detach()
        loss_l1 += F.l1_loss(lr_gan_rgb, lr_rgb).sum() * level_ratio * 0.25

        level_ratio = 1.0 if generator_level >= 1 else 0.1
        loss_p += (
            self.perceptual_loss(
                out["comp_gan_rgb"].permute(0, 3, 1, 2).contiguous(),
                gt_rgb.permute(0, 3, 1, 2).contiguous(),
            ).sum()
            * level_ratio
        )

        guidance_out = {
            "loss_l1": loss_l1,
            "loss_p": loss_p,
            "loss_G": loss_G,
            "loss_kl": loss_kl,
        }

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

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.manual_backward(loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        loss_D = discriminator_loss(
            self.renderer.discriminator,
            gt_rgb.permute(0, 3, 1, 2),
            out["comp_gan_rgb"].permute(0, 3, 1, 2),
        )
        loss_D *= self.C(self.cfg.loss["lambda_D"])
        self.log("train/loss_D", loss_D)
        self.manual_backward(loss_D)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

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
            f"it{self.true_global_step}-{batch['index'][0]}.jpg",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["comp_gan_rgb"][0],
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
                    "img": out["comp_gan_rgb"][0],
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

    def configure_optimizers(self):
        optimizer_g = parse_optimizer(self.cfg.optimizer, self)
        optimizer_d = parse_optimizer(self.cfg.optimizer.optimizer_dis, self)
        return [optimizer_g, optimizer_d], []
