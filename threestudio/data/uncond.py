import os
import json
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

from threestudio import register
from threestudio.utils.typing import *
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import get_ray_directions, get_rays, get_projection_matrix, get_mvp_matrix


@dataclass
class RandomCameraDataModuleConfig:
    height: int = 64
    width: int = 64
    eval_height: int = 512
    eval_width: int = 512
    batch_size: int = 1
    eval_batch_size: int = 1
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (40, 70) # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 0.
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 60.

class RandomCameraIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.directions_unit_focal = get_ray_directions(H=self.cfg.height, W=self.cfg.width, focal=1.0)
    
    def __iter__(self):
        while True:
            yield {}
    
    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles from a uniform distribution bounded by elevation_range
        elevation_deg: Float[Tensor, "B"] = torch.rand(self.cfg.batch_size) * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0]) + self.cfg.elevation_range[0]
        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        # azimuth_deg: Float[Tensor, "B"] = torch.rand(self.cfg.batch_size) * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]) + self.cfg.azimuth_range[0]
        azimuth_deg: Float[Tensor, "B"] = (torch.rand(self.cfg.batch_size) + torch.arange(self.cfg.batch_size)) / self.cfg.batch_size * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]) + self.cfg.azimuth_range[0]
        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = torch.rand(self.cfg.batch_size) * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0]) + self.cfg.camera_distance_range[0]

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack([
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation)
        ], dim=-1)

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None,:].repeat(self.cfg.batch_size, 1)

        # sample camera perturbations from a normal distribution with mean 0 and std camera_perturb
        camera_perturb: Float[Tensor, "B 3"] = torch.randn(self.cfg.batch_size, 3) * self.cfg.camera_perturb
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = torch.randn(self.cfg.batch_size, 3) * self.cfg.center_perturb
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = torch.randn(self.cfg.batch_size, 3) * self.cfg.up_perturb
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = torch.rand(self.cfg.batch_size) * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0]) + self.cfg.fovy_range[0]
        fovy = fovy_deg * math.pi / 180
        # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
        light_direction: Float[Tensor, "B 3"] = camera_positions + torch.randn(self.cfg.batch_size, 3) * self.cfg.light_position_perturb
        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = torch.rand(self.cfg.batch_size) * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0]) + self.cfg.light_distance_range[0]
        # get light position by scaling light direction by light distance
        light_positions: Float[Tensor, "B 3"] = F.normalize(light_direction, dim=-1) * light_distances[:,None]

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w: Float[Tensor, "B 3 4"] = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:,:,None]], dim=-1)

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.cfg.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[None,:,:,:].repeat(self.cfg.batch_size, 1, 1, 1)
        directions[:,:,:,:2] = directions[:,:,:,:2] / focal_length[:,None,None,None]

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.cfg.width / self.cfg.height, 0.1, 1000.) # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'mvp_mtx': mvp_mtx,
            'camera_positions': camera_positions,
            'light_positions': light_positions,
            'elevation': elevation_deg,
            'azimuth': azimuth_deg,
            'camera_distances': camera_distances,
            'height': self.cfg.height,
            'width': self.cfg.width
        }



class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == 'val':
            self.n_test_views = 5
        else:
            self.n_test_views = 120
        
        azimuth_deg: Float[Tensor, "B"] = torch.linspace(0, 360., self.n_test_views) - 180.
        elevation_deg: Float[Tensor, "B"] = torch.full_like(azimuth_deg, self.cfg.eval_elevation_deg)
        camera_distances: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.cfg.eval_camera_distance)

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack([
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation)
        ], dim=-1)

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None,:].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.cfg.eval_fovy_deg)
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w: Float[Tensor, "B 3 4"] = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:,:,None]], dim=-1)

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        directions_unit_focal = get_ray_directions(H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0)
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[None,:,:,:].repeat(self.n_test_views, 1, 1, 1)
        directions[:,:,:,:2] = directions[:,:,:,:2] / focal_length[:,None,None,None]

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.cfg.width / self.cfg.height, 0.1, 1000.) # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.camera_distances = camera_distances
    
    def __len__(self):
        return self.n_test_views
    
    def __getitem__(self, index):
        return {
            'rays_o': self.rays_o[index],
            'rays_d': self.rays_d[index],
            'mvp_mtx': self.mvp_mtx[index], 
            'camera_positions': self.camera_positions[index],
            'light_positions': self.light_positions[index],
            'elevation': self.elevation[index],
            'azimuth': self.azimuth[index],
            'camera_distances': self.camera_distances[index],
            'height': self.cfg.eval_height,
            'width': self.cfg.eval_width
        }

@register('random-camera-datamodule')
class RandomCameraDataModule(pl.LightningDataModule):

    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)
    
    def setup(self, stage=None) -> None:
        if stage in [None, 'fit']:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = RandomCameraDataset(self.cfg, 'val')
        if stage in [None, 'test']:
            self.test_dataset = RandomCameraDataset(self.cfg, 'test')

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, 
            num_workers=batch_size, # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.general_loader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.train_dataset.collate)

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1) 
