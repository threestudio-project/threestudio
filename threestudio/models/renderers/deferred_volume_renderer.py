from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.renderers.base import VolumeRenderer


class DeferredVolumeRenderer(VolumeRenderer):
    pass
