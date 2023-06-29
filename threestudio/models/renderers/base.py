from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class Renderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        # keep references to submodules using namedtuple, avoid being registered as modules
        @dataclass
        class SubModules:
            geometry: BaseImplicitGeometry
            material: BaseMaterial
            background: BaseBackground

        self.sub_modules = SubModules(geometry, material, background)

        # set up bounding box
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def geometry(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry

    @property
    def material(self) -> BaseMaterial:
        return self.sub_modules.material

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def set_geometry(self, geometry: BaseImplicitGeometry) -> None:
        self.sub_modules.geometry = geometry

    def set_material(self, material: BaseMaterial) -> None:
        self.sub_modules.material = material

    def set_background(self, background: BaseBackground) -> None:
        self.sub_modules.background = background


class VolumeRenderer(Renderer):
    pass


class Rasterizer(Renderer):
    pass
