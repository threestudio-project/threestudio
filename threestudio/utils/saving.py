import json
import os
import re
import shutil

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from threestudio.utils.typing import *


class SaverMixin:
    _save_dir: Optional[str] = None

    def set_save_dir(self, save_dir: str):
        self._save_dir = save_dir

    def get_save_dir(self):
        if self._save_dir is None:
            raise ValueError("Save dir is not set")
        return self._save_dir

    def convert_data(self, data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return [self.convert_data(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.convert_data(v) for k, v in data.items()}
        else:
            raise TypeError(
                "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
                type(data),
            )

    def get_save_path(self, filename):
        save_path = os.path.join(self.get_save_dir(), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    DEFAULT_RGB_KWARGS = {"data_format": "CHW", "data_range": (0, 1)}
    DEFAULT_UV_KWARGS = {
        "data_format": "CHW",
        "data_range": (0, 1),
        "cmap": "checkerboard",
    }
    DEFAULT_GRAYSCALE_KWARGS = {"data_range": None, "cmap": "jet"}

    def get_rgb_image_(self, img, data_format, data_range, rgba=False):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = img.clip(min=data_range[0], max=data_range[1])
            img = (
                (img - data_range[0]) / (data_range[1] - data_range[0]) * 255.0
            ).astype(np.uint8)
        nc = 4 if rgba else 3
        imgs = [img[..., start : start + nc] for start in range(0, img.shape[-1], nc)]
        imgs = [
            img_
            if img_.shape[-1] == nc
            else np.concatenate(
                [
                    img_,
                    np.zeros(
                        (img_.shape[0], img_.shape[1], nc - img_.shape[2]),
                        dtype=img_.dtype,
                    ),
                ],
                axis=-1,
            )
            for img_ in imgs
        ]
        img = np.concatenate(imgs, axis=1)
        if rgba:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def save_rgb_image(
        self,
        filename,
        img,
        data_format=DEFAULT_RGB_KWARGS["data_format"],
        data_range=DEFAULT_RGB_KWARGS["data_range"],
    ):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_uv_image_(self, img, data_format, data_range, cmap):
        img = self.convert_data(img)
        assert data_format in ["CHW", "HWC"]
        if data_format == "CHW":
            img = img.transpose(1, 2, 0)
        img = img.clip(min=data_range[0], max=data_range[1])
        img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in ["checkerboard", "color"]
        if cmap == "checkerboard":
            n_grid = 64
            mask = (img * n_grid).astype(int)
            mask = (mask[..., 0] + mask[..., 1]) % 2 == 0
            img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            img[mask] = np.array([255, 0, 255], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cmap == "color":
            img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_[..., 0] = (img[..., 0] * 255).astype(np.uint8)
            img_[..., 1] = (img[..., 1] * 255).astype(np.uint8)
            img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
            img = img_
        return img

    def save_uv_image(
        self,
        filename,
        img,
        data_format=DEFAULT_UV_KWARGS["data_format"],
        data_range=DEFAULT_UV_KWARGS["data_range"],
        cmap=DEFAULT_UV_KWARGS["cmap"],
    ):
        img = self.get_uv_image_(img, data_format, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_grayscale_image_(self, img, data_range, cmap):
        img = self.convert_data(img)
        img = np.nan_to_num(img)
        if data_range is None:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img.clip(data_range[0], data_range[1])
            img = (img - data_range[0]) / (data_range[1] - data_range[0])
        assert cmap in [None, "jet", "magma", "spectral"]
        if cmap == None:
            img = (img * 255.0).astype(np.uint8)
            img = np.repeat(img[..., None], 3, axis=2)
        elif cmap == "jet":
            img = (img * 255.0).astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        elif cmap == "magma":
            img = 1.0 - img
            base = cm.get_cmap("magma")
            num_bins = 256
            colormap = LinearSegmentedColormap.from_list(
                f"{base.name}{num_bins}", base(np.linspace(0, 1, num_bins)), num_bins
            )(np.linspace(0, 1, num_bins))[:, :3]
            a = np.floor(img * 255.0)
            b = (a + 1).clip(max=255.0)
            f = img * 255.0 - a
            a = a.astype(np.uint16).clip(0, 255)
            b = b.astype(np.uint16).clip(0, 255)
            img = colormap[a] + (colormap[b] - colormap[a]) * f[..., None]
            img = (img * 255.0).astype(np.uint8)
        elif cmap == "spectral":
            colormap = plt.get_cmap("Spectral")

            def blend_rgba(image):
                image = image[..., :3] * image[..., -1:] + (
                    1.0 - image[..., -1:]
                )  # blend A to RGB
                return image

            img = colormap(img)
            img = blend_rgba(img)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def save_grayscale_image(
        self,
        filename,
        img,
        data_range=DEFAULT_GRAYSCALE_KWARGS["data_range"],
        cmap=DEFAULT_GRAYSCALE_KWARGS["cmap"],
    ):
        img = self.get_grayscale_image_(img, data_range, cmap)
        cv2.imwrite(self.get_save_path(filename), img)

    def get_image_grid_(self, imgs):
        if isinstance(imgs[0], list):
            return np.concatenate([self.get_image_grid_(row) for row in imgs], axis=0)
        cols = []
        for col in imgs:
            assert col["type"] in ["rgb", "uv", "grayscale"]
            if col["type"] == "rgb":
                rgb_kwargs = self.DEFAULT_RGB_KWARGS.copy()
                rgb_kwargs.update(col["kwargs"])
                cols.append(self.get_rgb_image_(col["img"], **rgb_kwargs))
            elif col["type"] == "uv":
                uv_kwargs = self.DEFAULT_UV_KWARGS.copy()
                uv_kwargs.update(col["kwargs"])
                cols.append(self.get_uv_image_(col["img"], **uv_kwargs))
            elif col["type"] == "grayscale":
                grayscale_kwargs = self.DEFAULT_GRAYSCALE_KWARGS.copy()
                grayscale_kwargs.update(col["kwargs"])
                cols.append(self.get_grayscale_image_(col["img"], **grayscale_kwargs))
        return np.concatenate(cols, axis=1)

    def save_image_grid(self, filename, imgs):
        img = self.get_image_grid_(imgs)
        cv2.imwrite(self.get_save_path(filename), img)

    def save_image(self, filename, img):
        img = self.convert_data(img)
        assert img.dtype == np.uint8 or img.dtype == np.uint16
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(self.get_save_path(filename), img)

    def save_cubemap(self, filename, img, data_range=(0, 1), rgba=False):
        img = self.convert_data(img)
        assert img.ndim == 4 and img.shape[0] == 6 and img.shape[1] == img.shape[2]

        imgs_full = []
        for start in range(0, img.shape[-1], 3):
            img_ = img[..., start : start + 3]
            img_ = np.stack(
                [
                    self.get_rgb_image_(img_[i], "HWC", data_range, rgba=rgba)
                    for i in range(img_.shape[0])
                ],
                axis=0,
            )
            size = img_.shape[1]
            placeholder = np.zeros((size, size, 3), dtype=np.float32)
            img_full = np.concatenate(
                [
                    np.concatenate(
                        [placeholder, img_[2], placeholder, placeholder], axis=1
                    ),
                    np.concatenate([img_[1], img_[4], img_[0], img_[5]], axis=1),
                    np.concatenate(
                        [placeholder, img_[3], placeholder, placeholder], axis=1
                    ),
                ],
                axis=0,
            )
            imgs_full.append(img_full)

        imgs_full = np.concatenate(imgs_full, axis=1)
        cv2.imwrite(self.get_save_path(filename), imgs_full)

    def save_data(self, filename, data):
        data = self.convert_data(data)
        if isinstance(data, dict):
            if not filename.endswith(".npz"):
                filename += ".npz"
            np.savez(self.get_save_path(filename), **data)
        else:
            if not filename.endswith(".npy"):
                filename += ".npy"
            np.save(self.get_save_path(filename), data)

    def save_state_dict(self, filename, data):
        torch.save(data, self.get_save_path(filename))

    def save_img_sequence(self, filename, img_dir, matcher, save_format="mp4", fps=30):
        assert save_format in ["gif", "mp4"]
        if not filename.endswith(save_format):
            filename += f".{save_format}"
        matcher = re.compile(matcher)
        img_dir = os.path.join(self.get_save_dir(), img_dir)
        imgs = []
        for f in os.listdir(img_dir):
            if matcher.search(f):
                imgs.append(f)
        imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
        imgs = [cv2.imread(os.path.join(img_dir, f)) for f in imgs]

        if save_format == "gif":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(
                self.get_save_path(filename), imgs, fps=fps, palettesize=256
            )
        elif save_format == "mp4":
            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            imageio.mimsave(self.get_save_path(filename), imgs, fps=fps)

    def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None):
        v_pos = self.convert_data(v_pos)
        t_pos_idx = self.convert_data(t_pos_idx)
        mesh = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx)
        mesh.export(self.get_save_path(filename))

    def save_file(self, filename, src_path):
        shutil.copyfile(src_path, self.get_save_path(filename))

    def save_json(self, filename, payload):
        with open(self.get_save_path(filename), "w") as f:
            f.write(json.dumps(payload))
