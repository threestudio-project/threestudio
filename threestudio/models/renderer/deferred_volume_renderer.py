from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.renderer.base import VolumeRenderer


class DeferredVolumeRenderer(VolumeRenderer):
    pass
