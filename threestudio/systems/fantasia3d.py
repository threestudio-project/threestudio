from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("fantasia3d-system")
class Fantasia3D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # not necessary, just for backward compatibility
        material_type: str = "no-material"  # unused
        material: dict = field(default_factory=lambda: {"n_output_dims": 0})
        background_type: str = "solid-color-background"  # unused
        background: dict = field(default_factory=dict)
        renderer_type: str = "nvdiff-rasterizer"
        renderer: dict = field(default_factory=dict)
        ##################################################

        latent_steps: int = 1000

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_normal=True, render_rgb=False)
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

        # initialize SDF
        # FIXME: what if using other geometry types?
        self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()

        if self.true_global_step < self.cfg.latent_steps:
            guidance_inp = torch.cat(
                [out["comp_normal"] * 2.0 - 1.0, out["opacity"]], dim=-1
            )
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=True
            )
        else:
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        loss_normal_consistency = out["mesh"].normal_consistency()
        self.log("train/loss_normal_consistency", loss_normal_consistency)
        loss += loss_normal_consistency * self.C(
            self.cfg.loss.lambda_normal_consistency
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
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
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
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
