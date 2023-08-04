from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.models.networks import HyperNet
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("att3d-system")
class ATT3D(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        use_hyper_net: bool = True
        hidden_dim: int = 8

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer, prompt_processor
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        if self.cfg.use_hyper_net:
            self.hypernet = HyperNet(
                self.prompt_processor.text_embeddings.flatten().shape[0],
                self.geometry.encoding.encoding.encoding.params.shape[0],
                self.cfg.hidden_dim
            )
        else:
            size = self.geometry.encoding.encoding.encoding.params.shape[0]
            self.hypernet = nn.Parameter(
                (torch.rand(size).cuda() * 2 - 1) / 10000, requires_grad=True
            )

    def from_hyper_net(self):
        if self.cfg.use_hyper_net:
            prompt_utils = self.prompt_processor()
            features = self.hypernet(
                prompt_utils.text_embeddings.flatten().float().contiguous()
            )
            self.geometry.encoding.encoding.from_hyper_net(features)
        else:
            self.geometry.encoding.encoding.from_hyper_net(self.hypernet)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):

        self.prompt_processor.update_text_embeddings()
        prompt_utils = self.prompt_processor()
        self.from_hyper_net()
        out = self(batch)
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

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        demon_num = min(2, self.prompt_processor.prompt_tot)
        pid = self.prompt_processor.prompt_id
        for ind in range(demon_num):
            self.prompt_processor.prompt_id = ind
            self.prompt_processor.update_text_embeddings(fix=True)
            self.from_hyper_net()
            out = self(batch)
            self.save_image_grid(
                f"val/{self.prompt_processor.prompt}/it{self.true_global_step}-{batch['index'][0]}.png",
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

        # resume, only update in training_step
        self.prompt_processor.prompt_id = pid
        self.prompt_processor.update_text_embeddings(fix=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.from_hyper_net()

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"{self.prompt_processor.prompt}/it{self.true_global_step}-test/{batch['index'][0]}.png",
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
            f"{self.prompt_processor.prompt}-it{self.true_global_step}-test",
            f"{self.prompt_processor.prompt}/it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
