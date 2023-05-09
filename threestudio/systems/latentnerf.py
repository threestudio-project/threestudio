from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.ops import ShapeLoss, binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("latentnerf-system")
class LatentNeRF(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-volume"
        geometry: dict = field(default_factory=dict)
        material_type: str = "nomaterial"
        material: dict = field(default_factory=dict)
        background_type: str = "solid-color-background"
        background: dict = field(default_factory=dict)
        renderer_type: str = "nerf-volume-renderer"
        renderer: dict = field(default_factory=dict)
        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        prompt_processor_type: str = "stable-diffusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)

        guide_shape: Optional[str] = None
        refinement: bool = False

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

        if self.training or not self.cfg.refinement:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if self.cfg.guide_shape is not None:
            self.shape_loss = ShapeLoss(self.cfg.guide_shape)

    def forward(self, batch: Dict[str, Any], decode: bool = False) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        out = {
            **render_out,
        }
        if decode:
            if self.cfg.refinement:
                out["decoded_rgb"] = out["comp_rgb"]
            else:
                out["decoded_rgb"] = self.guidance.decode_latents(
                    out["comp_rgb"].permute(0, 3, 1, 2)
                ).permute(0, 2, 3, 1)
        return out

    def on_fit_start(self) -> None:
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor, self.trainer
        )

    def training_step(self, batch, batch_idx):
        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)
        guidance_out = self.guidance(
            out["comp_rgb"], text_embeddings, rgb_as_latents=not self.cfg.refinement
        )

        loss = 0.0

        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

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

        if (
            self.cfg.guide_shape is not None
            and self.C(self.cfg.loss.lambda_shape) > 0
            and out["points"].shape[0] > 0
        ):
            loss_shape = self.shape_loss(out["points"], out["density"])
            self.log("train/loss_shape", loss_shape)
            loss += loss_shape * self.C(self.cfg.loss.lambda_shape)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
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
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch, decode=True)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
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
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
