from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("zero123-simple-system")
class Zero123Simple(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        out = self(batch["random_camera"])
        guidance_out = self.guidance(
            out["comp_rgb"],
            **batch["random_camera"],
            rgb_as_latents=False,
        )

        loss = 0.0

        for name, value in guidance_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
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

        if self.C(self.cfg.loss.lambda_normal_smoothness_2d) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smoothness loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            loss_normal_smoothness_2d = (
                normal[:, 1:, :, :] - normal[:, :-1, :, :]
            ).square().mean() + (
                normal[:, :, 1:, :] - normal[:, :, :-1, :]
            ).square().mean()
            self.log("trian/loss_normal_smoothness_2d", loss_normal_smoothness_2d)
            loss += loss_normal_smoothness_2d * self.C(
                self.cfg.loss.lambda_normal_smoothness_2d
            )

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if self.true_global_step % 50 == 0:
            self.save_image_grid(
                f"it{self.true_global_step}-train-t{int(guidance_out['timesteps'][0])}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": guidance_out["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "rgb" in guidance_out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": guidance_out["rgb_1step_orig"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb_1step_orig" in guidance_out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": guidance_out["rgb_multistep_orig"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb_multistep_orig" in guidance_out
                    else []
                ),
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
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
