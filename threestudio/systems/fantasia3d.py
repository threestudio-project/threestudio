from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("fantasia3d-system")
class Fantasia3D(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-sdf"
        geometry: dict = field(default_factory=lambda: {"n_feature_dims": 0})
        material_type: str = "no-material"  # unused
        material: dict = field(default_factory=lambda: {"n_output_dims": 0})
        background_type: str = "solid-color-background"  # unused
        background: dict = field(default_factory=dict)
        renderer_type: str = "nvdiff-rasterizer"
        renderer: dict = field(default_factory=dict)
        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)
        prompt_processor_type: str = "dreamfusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)

        latent_steps: int = 2500

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
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_normal=True, render_rgb=False)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        """
        Initialize guidance and prompt processor in this hook:
        (1) excluded from optimizer parameters (this hook executes after optimizer is initialized)
        (2) only used in training
        To avoid being saved to checkpoints, see on_save_checkpoint below.
        """
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        # initialize SDF
        # FIXME: what if using other geometry types?
        self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = 0.0

        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)

        if self.global_step < self.cfg.latent_steps:
            guidance_inp = torch.cat(
                [out["comp_normal"] * 2.0 - 1.0, out["opacity"]], dim=-1
            )
            guidance_out = self.guidance(
                guidance_inp, text_embeddings, rgb_as_latents=True
            )
        else:
            guidance_inp = out["comp_normal"] * 2.0 - 1.0
            guidance_out = self.guidance(
                guidance_inp, text_embeddings, rgb_as_latents=False
            )

        loss += guidance_out["sds"] * self.C(self.cfg.loss.lambda_sds)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-{batch_idx}.png",
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
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.global_step}-test/{batch_idx}.png",
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
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove stable diffusion weights
        # TODO: better way?
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k.split(".")[0] not in ["prompt_processor", "guidance"]
        }
        return super().on_save_checkpoint(checkpoint)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # debug use
        pass
        # from lightning.pytorch.utilities import grad_norm
        # norms = grad_norm(self.geometry, norm_type=2)
        # print(norms)
