from dataclasses import dataclass, field

import numpy as np
import torch

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.typing import *


@threestudio.register("sjc-system")
class ScoreJacobianChaining(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "volume-grid"
        geometry: dict = field(default_factory=dict)
        material_type: str = "no-material"
        material: dict = field(default_factory=dict)
        background_type: str = "textured-background"
        background: dict = field(default_factory=dict)
        renderer_type: str = "nerf-volume-renderer"
        renderer: dict = field(default_factory=dict)
        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        prompt_processor_type: str = "stable-diffusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def forward(self, batch: Dict[str, Any], decode: bool = False) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        out = {
            **render_out,
        }
        if decode:
            out["decoded_rgb"] = self.guidance.decode_latents(
                out["comp_rgb"].permute(0, 3, 1, 2), latent_height=128, latent_width=128
            ).permute(0, 2, 3, 1)
        return out

    def on_fit_start(self) -> None:
        """
        Initialize prompt processor in this hook:
        (1) excluded from optimizer parameters (this hook executes after optimizer is initialized)
        (2) only used in training
        To avoid being saved to checkpoints, see on_save_checkpoint below.
        """
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )

    def on_test_start(self) -> None:
        # check if guidance is initialized, such as when loading from checkpoint
        if not hasattr(self, "guidance"):
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)
        guidance_out = self.guidance(
            out["comp_rgb"], text_embeddings, rgb_as_latents=True
        )

        loss = 0.0
        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

        loss_emptiness = (
            self.C(self.cfg.loss.lambda_emptiness)
            * torch.log(1 + self.cfg.loss.emptiness_scale * out["weights"]).mean()
        )

        self.log("train/loss_emptiness", loss_emptiness)
        loss += loss_emptiness

        # About the depth loss, see https://github.com/pals-ttic/sjc/issues/21
        if self.C(self.cfg.loss.lambda_depth) > 0:
            _, h, w, _ = out["comp_rgb"].shape
            comp_depth = (out["depth"] + 10 * (1 - out["opacity"])).squeeze(-1)
            center_h = int(self.cfg.loss.center_ratio * h)
            center_w = int(self.cfg.loss.center_ratio * w)
            border_h = (h - center_h) // 2
            border_w = (h - center_w) // 2
            center_depth = comp_depth[
                ..., border_h : border_h + center_h, border_w : border_w + center_w
            ]
            center_depth_mean = center_depth.mean()
            border_depth_mean = (comp_depth.sum() - center_depth.sum()) / (
                h * w - center_h * center_w
            )
            log_input = center_depth_mean - border_depth_mean + 1e-12
            loss_depth = (
                torch.sign(log_input)
                * torch.log(log_input)
                * self.C(self.cfg.loss.lambda_depth)
            )

            self.log("train/loss_depth", loss_depth)
            loss += loss_depth

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def vis_depth(self, pred_depth):
        depth = pred_depth.detach().cpu().numpy()
        depth = np.log(1.0 + depth + 1e-12) / np.log(1 + 10.0)
        return depth

    def validation_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        comp_depth = out["depth"] + 10 * (1 - out["opacity"])  # 10 for background
        vis_depth = self.vis_depth(comp_depth.squeeze(-1))

        self.save_image_grid(
            f"it{self.true_global_step}-{batch_idx}.png",
            [
                {
                    "type": "rgb",
                    "img": out["decoded_rgb"][0],
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
                    "type": "grayscale",
                    "img": vis_depth[0],
                    "kwargs": {"cmap": "spectral", "data_range": (0, 1)},
                },
            ],
            align=512,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch_idx}.png",
            [
                {
                    "type": "rgb",
                    "img": out["decoded_rgb"][0],
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
            align=1024,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
