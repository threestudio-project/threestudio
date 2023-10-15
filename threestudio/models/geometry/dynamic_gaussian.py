#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_knn._C import distCUDA2
from torch import nn
from torch.nn import functional as F

import threestudio
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.geometry.gaussian import BasicPointCloud, GaussianModel
from threestudio.utils.typing import *


@threestudio.register("dynamic-gaussian")
class DynamicGaussianModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        gaussian: Optional[GaussianModel.Config] = None
        dynamic_flow_name: str = ""
        dynamic_flow_config: Optional[BaseGeometry.Config] = None

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.gaussian = threestudio.find("gaussian")(self.cfg.gaussian)
        self.dynamic_flow = threestudio.find(self.cfg.dynamic_flow_name)(
            self.cfg.dynamic_flow_config
        )

    def setup_functions(self):
        self.gaussian.setup_functions()

    @property
    def gaussian_optimizer(self):
        return self.gaussian.optimizer

    @property
    def dynamic_optimizer(self):
        return self.dynamic_flow.optimizer

    @property
    def get_scaling(self):
        return self.gaussian.scaling_activation(self.gaussian._scaling)

    @property
    def get_rotation(self):
        return self.gaussian.rotation_activation(self.gaussian._rotation)

    @property
    def get_xyz(self):
        return self.gaussian._xyz

    @property
    def get_features(self):
        features_dc = self.gaussian._features_dc
        features_rest = self.gaussian._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.gaussian.opacity_activation(self.gaussian._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.gaussian.covariance_activation(
            self.gaussian.get_scaling, scaling_modifier, self.gaussian._rotation
        )

    def oneupSHdegree(self):
        if self.gaussian.active_sh_degree < self.gaussian.max_sh_degree:
            self.gaussian.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.gaussian.create_from_pcd(pcd, spatial_lr_scale)

    def training_setup(self):
        self.gaussian.training_setup()

    def update_learning_rate(self, iteration):
        self.gaussian.update_learning_rate(iteration)

    def update_learning_rate_fine(self, iteration):
        self.gaussian.update_learning_rate_fine(iteration)

    def construct_list_of_attributes(self):
        self.gaussian.construct_list_of_attributes()

    def reset_opacity(self):
        self.gaussian.reset_opacity()

    def to(self, device="cpu"):
        self.gaussian.to(device)

    def replace_tensor_to_optimizer(self, tensor, name):
        return self.gaussian.replace_tensor_to_optimizer(tensor, name)

    def _prune_optimizer(self, mask):
        return self.gaussian._prune_optimizer(mask)

    def prune_points(self, mask):
        self.gaussian.prune_points(mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        return self.gaussian.cat_tensors_to_optimizer(tensors_dict)

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        self.gaussian.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        self.gaussian.densify_and_split(grads, grad_threshold, scene_extent, N)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        self.gaussian.densify_and_clone(grads, grad_threshold, scene_extent)

    def densify(self, max_grad, extent):
        self.gaussian.densify(max_grad, extent)

    def prune(self, min_opacity, extent, max_screen_size):
        self.gaussian.prune(min_opacity, extent, max_screen_size)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.gaussian.add_densification_stats(viewspace_point_tensor, update_filter)

    @torch.no_grad()
    def update_states(
        self,
        iteration,
        visibility_filter,
        radii,
        viewspace_point_tensor,
        extent,
    ):
        self.gaussian.update_states(
            iteration, visibility_filter, radii, viewspace_point_tensor, extent
        )
