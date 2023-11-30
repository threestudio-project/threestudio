from dataclasses import dataclass, field

import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@threestudio.register("mesh-guidance")
class MeshGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        geometry_type: str = ""
        geometry: dict = field(default_factory=dict)
        renderer_type: str = ""
        renderer: dict = field(default_factory=dict)
        material_type: str = ""
        material: dict = field(default_factory=dict)
        background_type: str = ""
        background: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading obj")
        geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=geometry,
            material=material,
            background=background,
        )
        threestudio.info(f"Loaded mesh!")

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        guide_rgb = self.renderer(**kwargs)

        guidance_out = {"loss_l1": F.l1_loss(rgb, guide_rgb["comp_rgb"])}
        return guidance_out
