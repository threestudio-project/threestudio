from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import pdb
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from torchmetrics import PearsonCorrCoef

@threestudio.register("magic123-system")
class Magic123(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        refinement: bool = False
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
            self.cfg.guidance_3d
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.pearson = PearsonCorrCoef().to(self.device)

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        out_input = self(batch)
        out = self(batch["random_camera"])
        gt_mask = batch["mask"]
        prompt_utils = self.prompt_processor()
        guidance_out = self.guidance(
            out["comp_rgb"],
            prompt_utils,
            **batch["random_camera"],
            rgb_as_latents=False,
        )
        guidance_3d_out = self.guidance_3d(
            out["comp_rgb"],
            **batch["random_camera"],
            rgb_as_latents=False,
        )

        loss = 0.0

        loss_rgb = F.mse_loss(
            out_input["comp_rgb"],
            batch["rgb"] * batch["mask"].float()
            + out_input["comp_rgb_bg"] * (1.0 - batch["mask"].float()),
        )
        self.log("train/loss_rgb", loss_rgb)
        loss += loss_rgb * self.C(self.cfg.loss.lambda_rgb)

        loss_mask = F.binary_cross_entropy(
            out_input["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
            batch["mask"].float(),
        )
        self.log("train/loss_mask", loss_mask)
        loss += loss_mask * self.C(self.cfg.loss.lambda_mask)

        # depth loss
        if self.C(self.cfg.loss.lambda_depth) > 0:
            valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
            valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
            with torch.no_grad():
                A = torch.cat(
                    [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                )  # [B, 2]
                X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                valid_gt_depth = A @ X  # [B, 1]
            loss_depth = F.mse_loss(valid_gt_depth, valid_pred_depth)
            self.log("train/loss_depth", loss_depth)
            loss += loss_depth * self.C(self.cfg.loss.lambda_depth)

        # relative depth loss
        # pdb.set_trace()
        if self.C(self.cfg.loss.lambda_depth_rel) > 0:
            valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
            valid_pred_depth = out["depth"][gt_mask]  # [B,]
            loss_depth_rel = (1 - self.pearson(valid_pred_depth, valid_gt_depth)) / 2.0
            self.log("train/loss_depth_rel", loss_depth_rel)
            loss += loss_depth_rel * self.C(self.cfg.loss.lambda_depth_rel)
        
        # pdb.set_trace()
        # normal loss
        if self.C(self.cfg.loss.lambda_normal) > 0:
            valid_gt_normal = (
                1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
            )  # [B, 3]
            valid_pred_normal = (
                2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
            )  # [B, 3]
            # valid_pred_normal = (
            #     1 - 2 * out["comp_normal"][gt_mask.squeeze(-1)]
            # )  # [B, 3]
            loss_normal = (1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean())
            self.log("train/loss_normal", loss_normal)
            loss += loss_normal * self.C(self.cfg.loss.lambda_normal)

        for name, value in guidance_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in guidance_3d_out.items():
            if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                self.log(f"train/{name}_3d", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_3d_")]
                )

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
        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

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
