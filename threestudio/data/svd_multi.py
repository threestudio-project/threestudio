import bisect
import glob
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.svd_uncond import (
    SVDCameraDataModuleConfig,
    SVDCameraIterableDataset,
)
from threestudio.data.uncond import RandomCameraDataset
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class SVDMultiImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    images_dir_path: str = ""
    use_random_camera: bool = True
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False


class SVDMultiImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SVDMultiImageDataModuleConfig = cfg

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        self.n_views = self.cfg.n_train_views

        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360.0, self.n_views + 1)[1:]
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.n_views, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        self.fovy = fovy_deg * math.pi / 180
        # light_positions: Float[Tensor, "B 3"] = camera_positions
        # light position is always at front camera
        light_positions: Float[Tensor, "B 3"] = camera_positions[-1].repeat(
            len(camera_positions), 1
        )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * self.fovy)
        )
        # directions_unit_focal = get_ray_directions(
        #     H=self.height, W=self.width, focal=1.0
        # )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        # self.set_rays()
        self.load_images()
        self.prev_height = self.height

    def load_image(self, image_path):
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4])
        rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        print(f"[INFO] svd image dataset: load image {image_path} {rgb.shape}")
        return rgb, mask

    def load_images(self):
        # load image
        assert os.path.exists(
            self.cfg.images_dir_path
        ), f"Could not find images dir path {self.cfg.images_dir_path}!"
        self.rgb, self.mask = [], []
        for image_path in glob.glob(os.path.join(self.cfg.images_dir_path, "*")):
            rgb, mask = self.load_image(image_path)
            self.rgb.append(rgb)
            self.mask.append(mask)
        self.depth = None
        self.normal = None

    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # self.focal_length = self.focal_lengths[size_ind]
        # self.set_rays()
        self.load_images()

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.height / torch.tan(0.5 * self.fovy)
        )
        # directions_unit_focal = get_ray_directions(
        #     H=self.height, W=self.width, focal=1.0
        # )
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, self.c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx


class SVDMultiImageIterableDataset(IterableDataset, SVDMultiImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "ref_depth": self.depth,
            "ref_normal": self.normal,
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SVDMultiImageDataset(Dataset, SVDMultiImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]


@register("svd-image-datamodule")
class SVDImageDataModule(pl.LightningDataModule):
    cfg: SVDMultiImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SVDMultiImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SVDMultiImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SVDMultiImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SVDMultiImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
