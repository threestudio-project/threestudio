import os
from dataclasses import dataclass

import numpy as np
import torch
import trimesh

import threestudio
from threestudio.models.geometry.base import BaseExplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *


@threestudio.register("obj-mesh")
class OBJMesh(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        shape_init: str = ""
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"

    cfg: Config

    def configure(self) -> None:
        super().configure()
        # Initialize custom mesh
        if self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            self.v_pos = torch.tensor(mesh.vertices, dtype=torch.float32).to(
                self.device
            )
            self.v_color = torch.tensor(
                mesh.visual.vertex_colors[:, :3] / 255, dtype=torch.float32
            ).to(self.device)
            self.t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

    def isosurface(self):
        return self
