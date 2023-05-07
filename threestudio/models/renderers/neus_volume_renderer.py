from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.typing import *


class NeuSVolumeRenderer(VolumeRenderer):
    pass
