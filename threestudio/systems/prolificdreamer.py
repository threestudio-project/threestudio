import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("prolificdreamer-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        camera_condition_type: str = "extrinsics"
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
            coarse_system_cfg: ProlificDreamer.Config = parse_structured(
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
        # FIXME: not necessary in validation/testing
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

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

    def training_step(self, batch, batch_idx):
        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)

        if self.cfg.camera_condition_type == "extrinsics":
            Rt = batch["c2w"]  # B x 3 x 4
            Rt = torch.cat([Rt, torch.zeros_like(Rt[:, :1])], dim=1)
            Rt[:, 3, 3] = 0.0
            camera_condition = Rt
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = batch["mvp_mtx"]
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        guidance_out = self.guidance(
            out["comp_rgb"], text_embeddings, camera_condition, rgb_as_latents=False
        )

        loss = 0.0

        loss_vsd = guidance_out["vsd"]
        self.log("train/loss_vsd", loss_vsd)
        loss += loss_vsd * self.C(self.cfg.loss.lambda_vsd)

        loss_lora = guidance_out["lora"]
        self.log("train/loss_lora", loss_lora)
        loss += loss_lora * self.C(self.cfg.loss.lambda_lora)

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
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
