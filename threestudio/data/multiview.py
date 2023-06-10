import math
import random
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
import cv2

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W

@dataclass
class MultiviewsDataModuleConfig:
    dataroot: str = "dataset/face"
    train_downsample_resolution: int = 3 # 2^3
    eval_downsample_resolution: int = 3 # 2^3
    train_data_interval: int = 1
    eval_data_interval: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    light_sample_strategy: str = "dreamfusion"


class MultiviewIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        
        assert self.cfg.batch_size == 1
        scale = 2 ** self.cfg.train_downsample_resolution
        
        camera_dict = json.load(open(os.path.join(self.cfg.dataroot, "transforms.json"), "r"))
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        frames = frames[::self.cfg.train_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        print("Loading frames...")
        self.n_frames = len(frames)
        for frame in tqdm(frames):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"] / scale
            intrinsic[1, 1] = frame["fl_y"] / scale
            intrinsic[0, 2] = frame["cx"] / scale
            intrinsic[1, 2] = frame["cy"] / scale
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(frame["transform_matrix"], dtype=torch.float32)

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.frame_w, self.frame_h))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h, self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False
            )

            extr_rot: Float[Tensor, "3 3"] = extrinsic[:3, :3]
            extr_trans: Float[Tensor, "3 1"] = extrinsic[:3, 3:]
            c2w_rot: Float[Tensor, "3 3"] = extr_rot.T
            c2w_trans: Float[Tensor, "3 1"] = - extr_rot.T @ extr_trans
            c2w: Float[Tensor, "4 4"] = torch.eye(4)
            c2w[:3, :3] = c2w_rot
            c2w[:3, 3:] = c2w_trans
            c2w = convert_pose(c2w)

            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            K = intrinsic
            proj = [
                [2*K[0, 0]/self.frame_w,  -2*K[0, 1]/self.frame_w, (self.frame_w - 2*K[0, 2])/self.frame_w, 0],
                [0, -2*K[1, 1]/self.frame_h, (self.frame_h - 2*K[1, 2])/self.frame_h,                       0],
                [0, 0, (-far - near)/(far - near), -2*far*near/(far - near)],
                [0, 0, -1, 0],
            ]
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
        print("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(frames_direction, dim=0)
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(self.frames_direction, self.frames_c2w, keepdim=True)
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.frames_c2w, self.frames_proj)
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(self.frames_position)
       
    def __iter__(self):
        while True:
            yield {}
    
    def collate(self, batch):
        index = torch.randint(0, self.n_frames, (1,)).item()
        return {
            "index": index,
            "rays_o": self.rays_o[index:index+1],
            "rays_d": self.rays_d[index:index+1],
            "mvp_mtx": self.mvp_mtx[index:index+1],
            "c2w": self.frames_c2w[index:index+1],
            "camera_positions": self.frames_position[index:index+1],
            "light_positions": self.light_positions[index:index+1],
            "gt_rgb": self.frames_img[index:index+1],
            "height": self.frame_h,
            "width": self.frame_w
        }



class MultiviewDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: MultiviewsDataModuleConfig = cfg
        
        assert self.cfg.eval_batch_size == 1
        scale = 2 ** self.cfg.eval_downsample_resolution
        
        camera_dict = json.load(open(os.path.join(self.cfg.dataroot, "transforms.json"), "r"))
        assert camera_dict["camera_model"] == "OPENCV"

        frames = camera_dict["frames"]
        frames = frames[::self.cfg.eval_data_interval]
        frames_proj = []
        frames_c2w = []
        frames_position = []
        frames_direction = []
        frames_img = []

        self.frame_w = frames[0]["w"] // scale
        self.frame_h = frames[0]["h"] // scale
        print("Loading frames...")
        self.n_frames = len(frames)
        for frame in tqdm(frames):
            intrinsic: Float[Tensor, "4 4"] = torch.eye(4)
            intrinsic[0, 0] = frame["fl_x"] / scale
            intrinsic[1, 1] = frame["fl_y"] / scale
            intrinsic[0, 2] = frame["cx"] / scale
            intrinsic[1, 2] = frame["cy"] / scale
            extrinsic: Float[Tensor, "4 4"] = torch.as_tensor(frame["transform_matrix"], dtype=torch.float32)

            frame_path = os.path.join(self.cfg.dataroot, frame["file_path"])
            img = cv2.imread(frame_path)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.frame_w, self.frame_h))
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            frames_img.append(img)

            direction: Float[Tensor, "H W 3"] = get_ray_directions(
                self.frame_h, self.frame_w,
                (intrinsic[0, 0], intrinsic[1, 1]),
                (intrinsic[0, 2], intrinsic[1, 2]),
                use_pixel_centers=False
            )

            extr_rot: Float[Tensor, "3 3"] = extrinsic[:3, :3]
            extr_trans: Float[Tensor, "3 1"] = extrinsic[:3, 3:]
            c2w_rot: Float[Tensor, "3 3"] = extr_rot.T
            c2w_trans: Float[Tensor, "3 1"] = - extr_rot.T @ extr_trans
            c2w: Float[Tensor, "4 4"] = torch.eye(4)
            c2w[:3, :3] = c2w_rot
            c2w[:3, 3:] = c2w_trans
            c2w = convert_pose(c2w)

            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)

            near = 0.1
            far = 1000.0
            K = intrinsic
            proj = [
                [2*K[0, 0]/self.frame_w,  -2*K[0, 1]/self.frame_w, (self.frame_w - 2*K[0, 2])/self.frame_w, 0],
                [0, -2*K[1, 1]/self.frame_h, (self.frame_h - 2*K[1, 2])/self.frame_h,                       0],
                [0, 0, (-far - near)/(far - near), -2*far*near/(far - near)],
                [0, 0, -1, 0],
            ]
            proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
            frames_proj.append(proj)
            frames_c2w.append(c2w)
            frames_position.append(camera_position)
            frames_direction.append(direction)
        print("Loaded frames.")

        self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        self.frames_c2w: Float[Tensor, "B 4 4"] = torch.stack(frames_c2w, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(frames_position, dim=0)
        self.frames_direction: Float[Tensor, "B H W 3"] = torch.stack(frames_direction, dim=0)
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(frames_img, dim=0)

        self.rays_o, self.rays_d = get_rays(self.frames_direction, self.frames_c2w, keepdim=True)
        self.mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.frames_c2w, self.frames_proj)
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(self.frames_position)
            
    def __len__(self):
        return self.n_frames

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index]
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        return batch


@register("multiview-camera-datamodule")
class MultiviewDataModule(pl.LightningDataModule):
    cfg: MultiviewsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiviewIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiviewDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiviewDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
