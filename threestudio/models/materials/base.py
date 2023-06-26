import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config
    requires_normal: bool = False
    requires_tangent: bool = False

    def configure(self):
        pass

    def forward(self, *args, **kwargs) -> Float[Tensor, "*B 3"]:
        raise NotImplementedError

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}
