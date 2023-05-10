import gzip
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
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


def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def _load_depth(path, scale_adjustment) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


# Code adapted from https://github.com/eldar/snes/blob/473ff2b1f6/3rdparty/co3d/dataset/co3d_dataset.py
def _get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def get_bbox_from_mask(mask, thr, decrease_quant=0.05):
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def get_clamp_bbox(bbox, box_crop_context=0.0, impath=""):
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.astype(np.float32)
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        warnings.warn(f"squashed image {impath}!!")
        return None

    # bbox[2:] = np.clip(bbox[2:], 2, )
    bbox[2:] = np.maximum(bbox[2:], 2)
    bbox[2:] += bbox[0:2] + 1  # convert to [xmin, ymin, xmax, ymax]
    # +1 because upper bound is not inclusive

    return bbox


def crop_around_box(tensor, bbox, impath=""):
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, tensor.shape[-2])
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, tensor.shape[-3])
    bbox = bbox.round().astype(np.longlong)
    return tensor[bbox[1] : bbox[3], bbox[0] : bbox[2], ...]


def resize_image(image, height, width, mode="bilinear"):
    if image.shape[:2] == (height, width):
        return image, 1.0, np.ones_like(image[..., :1])

    image = torch.from_numpy(image).permute(2, 0, 1)
    minscale = min(height / image.shape[-2], width / image.shape[-1])
    imre = torch.nn.functional.interpolate(
        image[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]

    # pyre-fixme[19]: Expected 1 positional argument.
    imre_ = torch.zeros(image.shape[0], height, width)
    imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
    # pyre-fixme[6]: For 2nd param expected `int` but got `Optional[int]`.
    # pyre-fixme[6]: For 3rd param expected `int` but got `Optional[int]`.
    mask = torch.zeros(1, height, width)
    mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
    return imre_.permute(1, 2, 0).numpy(), minscale, mask.permute(1, 2, 0).numpy()


# Code adapted from https://github.com/POSTECH-CVLab/PeRFception/data_util/co3d.py
def similarity_from_cameras(c2w, fix_rot=False, radius=1.0):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, 0.0, 1.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if fix_rot:
        R_align = np.eye(3)
        R = np.eye(3)
    else:
        R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = radius / np.median(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


@dataclass
class Co3dDataModuleConfig:
    root_dir: str = ""
    batch_size: int = 1
    height: int = 256
    width: int = 256
    load_preprocessed: bool = False
    cam_scale_factor: float = 0.95
    max_num_frames: int = 300
    v2_mode: bool = True
    use_mask: bool = True
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    train_num_rays: int = -1
    train_views: Optional[list] = None
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    scale_radius: float = 1.0
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 0.0
    render_path: str = "circle"


class Co3dDatasetBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: Co3dDataModuleConfig = cfg

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

        self.use_mask = self.cfg.use_mask
        cam_scale_factor = self.cfg.cam_scale_factor

        assert os.path.exists(self.cfg.root_dir), f"{self.cfg.root_dir} doesn't exist!"

        cam_trans = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
        scene_number = self.cfg.root_dir.split("/")[-1]
        json_path = os.path.join(self.cfg.root_dir, "..", "frame_annotations.jgz")
        with gzip.open(json_path, "r") as fp:
            all_frames_data = json.load(fp)

        frame_data, images, intrinsics, extrinsics, image_sizes = [], [], [], [], []
        masks = []
        depths = []

        for temporal_data in all_frames_data:
            if temporal_data["sequence_name"] == scene_number:
                frame_data.append(temporal_data)

        self.all_directions = []
        self.all_fg_masks = []
        for frame in frame_data:
            if "unseen" in frame["meta"]["frame_type"]:
                continue
            img = cv2.imread(
                os.path.join(self.cfg.root_dir, "..", "..", frame["image"]["path"])
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            # TODO: use estimated depth
            depth = _load_depth(
                os.path.join(self.cfg.root_dir, "..", "..", frame["depth"]["path"]),
                frame["depth"]["scale_adjustment"],
            )[0]

            H, W = frame["image"]["size"]
            image_size = np.array([H, W])
            fxy = np.array(frame["viewpoint"]["focal_length"])
            cxy = np.array(frame["viewpoint"]["principal_point"])
            R = np.array(frame["viewpoint"]["R"])
            T = np.array(frame["viewpoint"]["T"])

            if self.cfg.v2_mode:
                min_HW = min(W, H)
                image_size_half = np.array([W * 0.5, H * 0.5], dtype=np.float32)
                scale_arr = np.array([min_HW * 0.5, min_HW * 0.5], dtype=np.float32)
                fxy_x = fxy * scale_arr
                prp_x = np.array([W * 0.5, H * 0.5], dtype=np.float32) - cxy * scale_arr
                cxy = (image_size_half - prp_x) / image_size_half
                fxy = fxy_x / image_size_half

            scale_arr = np.array([W * 0.5, H * 0.5], dtype=np.float32)
            focal = fxy * scale_arr
            prp = -1.0 * (cxy - 1.0) * scale_arr

            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3:] = -R @ T[..., None]
            # original camera: x left, y up, z in (Pytorch3D)
            # transformed camera: x right, y down, z in (OpenCV)
            pose = pose @ cam_trans
            intrinsic = np.array(
                [
                    [focal[0], 0.0, prp[0], 0.0],
                    [0.0, focal[1], prp[1], 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            if any([np.all(pose == _pose) for _pose in extrinsics]):
                continue

            image_sizes.append(image_size)
            intrinsics.append(intrinsic)
            extrinsics.append(pose)
            images.append(img)
            depths.append(depth)
            self.all_directions.append(get_ray_directions(W, H, focal, prp))

            # vis_utils.vis_depth_pcd([depth], [pose], intrinsic, [(img * 255).astype(np.uint8)])

            if self.use_mask:
                mask = np.array(
                    Image.open(
                        os.path.join(
                            self.cfg.root_dir, "..", "..", frame["mask"]["path"]
                        )
                    )
                )
                mask = mask.astype(np.float32) / 255.0  # (h, w)
            else:
                mask = torch.ones_like(img[..., 0])
            self.all_fg_masks.append(mask)

        intrinsics = np.stack(intrinsics)
        extrinsics = np.stack(extrinsics)
        image_sizes = np.stack(image_sizes)
        self.all_directions = torch.stack(self.all_directions, dim=0)
        self.all_fg_masks = np.stack(self.all_fg_masks, 0)

        H_median, W_median = np.median(
            np.stack([image_size for image_size in image_sizes]), axis=0
        )

        H_inlier = np.abs(image_sizes[:, 0] - H_median) / H_median < 0.1
        W_inlier = np.abs(image_sizes[:, 1] - W_median) / W_median < 0.1
        inlier = np.logical_and(H_inlier, W_inlier)
        dists = np.linalg.norm(
            extrinsics[:, :3, 3] - np.median(extrinsics[:, :3, 3], axis=0), axis=-1
        )
        med = np.median(dists)
        good_mask = dists < (med * 5.0)
        inlier = np.logical_and(inlier, good_mask)

        if inlier.sum() != 0:
            intrinsics = intrinsics[inlier]
            extrinsics = extrinsics[inlier]
            image_sizes = image_sizes[inlier]
            images = [images[i] for i in range(len(inlier)) if inlier[i]]
            depths = [depths[i] for i in range(len(inlier)) if inlier[i]]
            self.all_directions = self.all_directions[inlier]
            self.all_fg_masks = self.all_fg_masks[inlier]

        extrinsics = np.stack(extrinsics)
        T, sscale = similarity_from_cameras(extrinsics, radius=self.cfg.scale_radius)
        extrinsics = T @ extrinsics

        extrinsics[:, :3, 3] *= sscale * cam_scale_factor

        depths = [depth * sscale * cam_scale_factor for depth in depths]

        num_frames = len(extrinsics)

        if self.cfg.max_num_frames < num_frames:
            num_frames = self.cfg.max_num_frames
            extrinsics = extrinsics[:num_frames]
            intrinsics = intrinsics[:num_frames]
            image_sizes = image_sizes[:num_frames]
            images = images[:num_frames]
            depths = depths[:num_frames]
            self.all_directions = self.all_directions[:num_frames]
            self.all_fg_masks = self.all_fg_masks[:num_frames]

        if self.cfg.box_crop:
            print("cropping...")
            crop_masks = []
            crop_imgs = []
            crop_depths = []
            crop_directions = []
            crop_xywhs = []
            max_sl = 0
            for i in range(num_frames):
                bbox_xywh = np.array(
                    get_bbox_from_mask(self.all_fg_masks[i], self.cfg.box_crop_mask_thr)
                )
                clamp_bbox_xywh = get_clamp_bbox(bbox_xywh, self.cfg.box_crop_context)
                max_sl = max(clamp_bbox_xywh[2] - clamp_bbox_xywh[0], max_sl)
                max_sl = max(clamp_bbox_xywh[3] - clamp_bbox_xywh[1], max_sl)
                mask = crop_around_box(self.all_fg_masks[i][..., None], clamp_bbox_xywh)
                img = crop_around_box(images[i], clamp_bbox_xywh)
                depth = crop_around_box(depths[i][..., None], clamp_bbox_xywh)

                # resize to the same shape
                mask, _, _ = resize_image(mask, self.cfg.height, self.cfg.width)
                depth, _, _ = resize_image(depth, self.cfg.height, self.cfg.width)
                img, scale, _ = resize_image(img, self.cfg.height, self.cfg.width)
                fx, fy, cx, cy = (
                    intrinsics[i][0, 0],
                    intrinsics[i][1, 1],
                    intrinsics[i][0, 2],
                    intrinsics[i][1, 2],
                )

                crop_masks.append(mask)
                crop_imgs.append(img)
                crop_depths.append(depth)
                crop_xywhs.append(clamp_bbox_xywh)
                crop_directions.append(
                    get_ray_directions(
                        self.cfg.height,
                        self.cfg.width,
                        (fx * scale, fy * scale),
                        (
                            (cx - clamp_bbox_xywh[0]) * scale,
                            (cy - clamp_bbox_xywh[1]) * scale,
                        ),
                    )
                )

            # # pad all images to the same shape
            # for i in range(num_frames):
            #     uh = (max_sl - crop_imgs[i].shape[0]) // 2 # h
            #     dh = max_sl - crop_imgs[i].shape[0] - uh
            #     lw = (max_sl - crop_imgs[i].shape[1]) // 2
            #     rw = max_sl - crop_imgs[i].shape[1] - lw
            #     crop_masks[i] = np.pad(crop_masks[i], pad_width=((uh, dh), (lw, rw), (0, 0)), mode='constant', constant_values=0.)
            #     crop_imgs[i] = np.pad(crop_imgs[i], pad_width=((uh, dh), (lw, rw), (0, 0)), mode='constant', constant_values=1.)
            #     crop_depths[i] = np.pad(crop_depths[i], pad_width=((uh, dh), (lw, rw), (0, 0)), mode='constant', constant_values=0.)
            #     fx, fy, cx, cy = intrinsics[i][0, 0], intrinsics[i][1, 1], intrinsics[i][0, 2], intrinsics[i][1, 2]
            #     crop_directions.append(get_ray_directions(max_sl, max_sl, (fx, fy), (cx - crop_xywhs[i][0] + lw, cy - crop_xywhs[i][1] + uh)))
            # self.w, self.h = max_sl, max_sl

            images = crop_imgs
            depths = crop_depths
            self.all_fg_masks = np.stack(crop_masks, 0)
            self.all_directions = torch.from_numpy(np.stack(crop_directions, 0))

        # self.width, self.height = self.w, self.h

        self.all_c2w = torch.from_numpy(
            (
                extrinsics
                @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))[None, ...]
            )[..., :3, :4]
        )
        self.all_images = torch.from_numpy(np.stack(images, axis=0))
        self.all_depths = torch.from_numpy(np.stack(depths, axis=0))

        # self.all_c2w = []
        # self.all_images = []
        # for i in range(num_frames):
        #     # convert to: x right, y up, z back (OpenGL)
        #     c2w = torch.from_numpy(extrinsics[i] @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32)))[:3, :4]
        #     self.all_c2w.append(c2w)
        #     img = torch.from_numpy(images[i])
        #     self.all_images.append(img)

        # TODO: save data for fast loading next time
        if self.cfg.load_preprocessed and os.path.exists(
            self.cfg.root_dir, "nerf_preprocessed.npy"
        ):
            pass

        i_all = np.arange(num_frames)

        if self.cfg.train_views is None:
            i_test = i_all[::10]
            i_val = i_test
            i_train = np.array([i for i in i_all if not i in i_test])
        else:
            # use provided views
            i_train = self.cfg.train_views
            i_test = np.array([i for i in i_all if not i in i_train])
            i_val = i_test

        if self.split == "train":
            print("[INFO] num of train views: ", len(i_train))
            print("[INFO] train view ids = ", i_train)

        i_split = {"train": i_train, "val": i_val, "test": i_all}

        # if self.split == 'test':
        #     self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.cfg.n_test_traj_steps)
        #     self.all_images = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
        #     self.all_fg_masks = torch.zeros((self.cfg.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
        #     self.directions = self.directions[0].to(self.rank)
        # else:
        self.all_images, self.all_c2w = (
            self.all_images[i_split[self.split]],
            self.all_c2w[i_split[self.split]],
        )
        self.all_directions = self.all_directions[i_split[self.split]].to(self.rank)
        self.all_fg_masks = torch.from_numpy(self.all_fg_masks)[i_split[self.split]]
        self.all_depths = self.all_depths[i_split[self.split]]
        # if render_random_pose:
        #     render_poses = random_pose(extrinsics[i_all], 50)
        # elif render_scene_interp:
        #     render_poses = pose_interp(extrinsics[i_all], interp_fac)
        # render_poses = spherical_poses(sscale * cam_scale_factor * np.eye(4))

        # near, far = 0., 1.
        # ndc_coeffs = (-1., -1.)

        self.all_c2w, self.all_images, self.all_fg_masks = (
            self.all_c2w.float().to(self.rank),
            self.all_images.float().to(self.rank),
            self.all_fg_masks.float().to(self.rank),
        )

        # self.all_c2w, self.all_images, self.all_fg_masks = \
        #         self.all_c2w.float(), \
        #         self.all_images.float(), \
        #         self.all_fg_masks.float()

        self.all_depths = self.all_depths.float().to(self.rank)

    def get_all_images(self):
        return self.all_images


class Co3dDataset(Dataset, Co3dDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return len(self.random_pose_generator)
            else:
                return len(self.all_images)
        else:
            return len(self.random_pose_generator)
            # return len(self.all_images)

    def prepare_data(self, index):
        # prepare batch data here
        c2w = self.all_c2w[index]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[index]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb = self.all_images[index]
        depth = self.all_depths[index]
        mask = self.all_fg_masks[index]

        # TODO: get projection matrix and mvp matrix
        # proj_mtx = get_projection_matrix()

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": 0,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": 0,
            "azimuth": 0,
            "camera_distances": 0,
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
        }

        # c2w = self.all_c2w[index]
        # return {
        #     'index': index,
        #     'c2w': c2w,
        #     'light_positions': c2w[:3, -1],
        #     'H': self.h,
        #     'W': self.w
        # }

        return batch

    def __getitem__(self, index):
        if self.split == "test":
            if self.cfg.render_path == "circle":
                return self.random_pose_generator[index]
            else:
                return self.prepare_data(index)
        else:
            return self.random_pose_generator[index]


class Co3dIterableDataset(IterableDataset, Co3dDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)
        self.idx = 0
        self.image_perm = torch.randperm(len(self.all_images))

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        idx = self.image_perm[self.idx]
        # prepare batch data here
        c2w = self.all_c2w[idx][None]
        light_positions = c2w[..., :3, -1]
        directions = self.all_directions[idx][None]
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )
        rgb = self.all_images[idx][None]
        depth = self.all_depths[idx][None]
        mask = self.all_fg_masks[idx][None]

        if (
            self.cfg.train_num_rays != -1
            and self.cfg.train_num_rays < self.cfg.height * self.cfg.width
        ):
            _, height, width, _ = rays_o.shape
            x = torch.randint(
                0, width, size=(self.cfg.train_num_rays,), device=rays_o.device
            )
            y = torch.randint(
                0, height, size=(self.cfg.train_num_rays,), device=rays_o.device
            )

            rays_o = rays_o[:, y, x].unsqueeze(-2)
            rays_d = rays_d[:, y, x].unsqueeze(-2)
            directions = directions[:, y, x].unsqueeze(-2)
            rgb = rgb[:, y, x].unsqueeze(-2)
            mask = mask[:, y, x].unsqueeze(-2)
            depth = depth[:, y, x].unsqueeze(-2)

        # TODO: get projection matrix and mvp matrix
        # proj_mtx = get_projection_matrix()

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": None,
            "camera_positions": c2w[..., :3, -1],
            "light_positions": light_positions,
            "elevation": None,
            "azimuth": None,
            "camera_distances": None,
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
        }

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        # prepare batch data in system
        # c2w = self.all_c2w[idx][None]

        # batch = {
        #     'index': torch.tensor([idx]),
        #     'c2w': c2w,
        #     'light_positions': c2w[..., :3, -1],
        #     'H': self.h,
        #     'W': self.w
        # }

        self.idx += 1
        if self.idx == len(self.all_images):
            self.idx = 0
            self.image_perm = torch.randperm(len(self.all_images))
        # self.idx = (self.idx + 1) % len(self.all_images)

        return batch


@register("co3d-datamodule")
class Co3dDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(Co3dDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = Co3dIterableDataset(self.cfg, self.cfg.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = Co3dDataset(self.cfg, self.cfg.val_split)
        if stage in [None, "test", "predict"]:
            self.test_dataset = Co3dDataset(self.cfg, self.cfg.test_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        sampler = None
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            # pin_memory=True,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset, batch_size=1, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)
