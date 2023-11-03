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
from plyfile import PlyData
from simple_knn._C import distCUDA2
from torch import nn
from torch.nn import functional as F

import threestudio
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.geometry.gaussian import BasicPointCloud, GaussianModel
from threestudio.utils.GAN.discriminator import NLayerDiscriminator, weights_init
from threestudio.utils.GAN.normalunet import NormalNet
from threestudio.utils.typing import *


@threestudio.register("dynamic-gaussian")
class DynamicGaussianModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        gaussian_name: str = ""
        gaussian: Optional[BaseGeometry.Config] = None
        dynamic_flow_name: str = ""
        dynamic_flow_config: Optional[BaseGeometry.Config] = None

        geometry_convert_from: str = ""

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.gaussian = threestudio.find(self.cfg.gaussian_name)(self.cfg.gaussian)
        self.dynamic_flow = threestudio.find(self.cfg.dynamic_flow_name)(
            self.cfg.dynamic_flow_config
        )
        self.refine_net = NormalNet(ngf=32, n_downsampling=3, n_blocks=3)
        self.discriminator = NLayerDiscriminator(
            input_nc=3, n_layers=3, use_actnorm=False, ndf=64
        ).apply(weights_init)

        if len(self.cfg.geometry_convert_from) > 0:
            print("Loading point cloud from %s" % self.cfg.geometry_convert_from)
            if self.cfg.geometry_convert_from.endswith(".ckpt"):
                ckpt_dict = torch.load(self.cfg.geometry_convert_from)
                num_pts = ckpt_dict["state_dict"]["geometry.gaussian._xyz"].shape[0]
                pcd = BasicPointCloud(
                    points=np.zeros((num_pts, 3)),
                    colors=np.zeros((num_pts, 3)),
                    normals=np.zeros((num_pts, 3)),
                )
                self.gaussian.create_from_pcd(pcd, 10)
                self.gaussian.training_setup()
                new_ckpt_dict = {}
                for key in self.gaussian.state_dict():
                    if ckpt_dict["state_dict"].__contains__("geometry.gaussian." + key):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"][
                            "geometry.gaussian." + key
                        ]
                    else:
                        new_ckpt_dict[key] = self.gaussian.state_dict()[key]
                self.gaussian.load_state_dict(new_ckpt_dict)

                new_ckpt_dict = {}
                for key in self.dynamic_flow.state_dict():
                    if ckpt_dict["state_dict"].__contains__(
                        "geometry.dynamic_flow." + key
                    ):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"][
                            "geometry.dynamic_flow." + key
                        ]
                    else:
                        new_ckpt_dict[key] = self.dynamic_flow.state_dict()[key]
                self.dynamic_flow.load_state_dict(new_ckpt_dict)

                new_ckpt_dict = {}
                for key in self.refine_net.state_dict():
                    if ckpt_dict["state_dict"].__contains__(
                        "geometry.refine_net." + key
                    ):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"][
                            "geometry.refine_net." + key
                        ]
                    else:
                        new_ckpt_dict[key] = self.refine_net.state_dict()[key]
                self.refine_net.load_state_dict(new_ckpt_dict)

                new_ckpt_dict = {}
                for key in self.discriminator.state_dict():
                    if ckpt_dict["state_dict"].__contains__(
                        "geometry.discriminator." + key
                    ):
                        new_ckpt_dict[key] = ckpt_dict["state_dict"][
                            "geometry.discriminator." + key
                        ]
                    else:
                        new_ckpt_dict[key] = self.discriminator.state_dict()[key]
                self.discriminator.load_state_dict(new_ckpt_dict)
            elif self.cfg.geometry_convert_from.endswith(".ply"):
                plydata = PlyData.read(self.cfg.geometry_convert_from)
                vertices = plydata["vertex"]
                positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
                colors = (
                    np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T
                    / 255.0
                )
                normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
                # print(np.min(positions, axis=0))
                # print(np.max(positions, axis=0))
                # exit(0)
                pcd = BasicPointCloud(points=positions, colors=colors, normals=normals)
                self.gaussian.create_from_pcd(pcd, 10)
                self.gaussian.training_setup()

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
        return self.gaussian.get_scaling

    @property
    def get_rotation(self):
        return self.gaussian.get_rotation

    @property
    def get_xyz(self):
        return self.gaussian.get_xyz

    @property
    def get_features(self):
        return self.gaussian.get_features

    @property
    def get_opacity(self):
        return self.gaussian.get_opacity

    def get_covariance(self, scaling_modifier=1):
        return self.get_covariance(scaling_modifier)

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

    def save_ply(self, path):
        self.gaussian.save_ply(path)
