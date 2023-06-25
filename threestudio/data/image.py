import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
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
class SingleImageDataModuleConfig:
    height: int = 96
    width: int = 96
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1


class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        # load image
        assert os.path.exists(self.cfg.image_path)
        rgba = cv2.cvtColor(
            cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        print(
            f"[INFO] single image dataset: load image {self.cfg.image_path} {self.rgb.shape}"
        )

        # load depth
        depth_path = self.cfg.image_path.replace("_rgba.png", "_depth.png")
        assert os.path.exists(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(
            depth, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA
        )
        self.depth: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(depth.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(self.rank)
        )
        print(
            f"[INFO] single image dataset: load depth {depth_path} {self.depth.shape}"
        )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )

        # get directions by dividing directions_unit_focal by focal length
        fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))
        focal_length = 0.5 * self.cfg.height / torch.tan(0.5 * fovy)
        directions_unit_focal = get_ray_directions(
            H=self.cfg.height, W=self.cfg.width, focal=1.0
        )
        directions: Float[Tensor, "1 H W 3"] = directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length

        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            fovy, self.cfg.width / self.cfg.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance

    def get_all_images(self):
        return self.rgb


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase):
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
            "depth": self.depth,
            "mask": self.mask,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]
        # if index == 0:
        #     return {
        #         'rays_o': self.rays_o[0],
        #         'rays_d': self.rays_d[0],
        #         'mvp_mtx': self.mvp_mtx[0],
        #         'camera_positions': self.camera_position[0],
        #         'light_positions': self.light_position[0],
        #         'elevation': self.elevation_deg[0],
        #         'azimuth': self.azimuth_deg[0],
        #         'camera_distances': self.camera_distance[0],
        #         'rgb': self.rgb[0],
        #         'depth': self.depth[0],
        #         'mask': self.mask[0]
        #     }
        # else:
        #     return self.random_pose_generator[index - 1]


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

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
