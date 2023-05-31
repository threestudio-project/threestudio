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
from pytorch_lightning.loggers import WandbLogger

import wandb
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *


class SaverMixin:
    _save_dir: Optional[str] = None
    _wandb_logger: Optional[WandbLogger] = None

    def set_save_dir(self, save_dir: str):
        self._save_dir = save_dir

    def get_save_dir(self):
        if self._save_dir is None:
            raise ValueError("Save dir is not set")
        return self._save_dir

    def convert_data(self, data):
        if data is None:
            return None
        elif isinstance(data, np.ndarray):
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

    def create_loggers(self, cfg_loggers: DictConfig) -> None:
        if "wandb" in cfg_loggers.keys() and cfg_loggers.wandb.enable:
            self._wandb_logger = WandbLogger(project=cfg_loggers.wandb.project)

    def get_loggers(self) -> List:
        if self._wandb_logger:
            return [self._wandb_logger]
        else:
            return []

    DEFAULT_RGB_KWARGS = {"data_format": "HWC", "data_range": (0, 1)}
    DEFAULT_UV_KWARGS = {
        "data_format": "HWC",
        "data_range": (0, 1),
        "cmap": "checkerboard",
    }
    DEFAULT_GRAYSCALE_KWARGS = {"data_range": None, "cmap": "jet"}
    DEFAULT_GRID_KWARGS = {"align": "max"}

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

    def _save_rgb_image(
        self,
        filename,
        img,
        data_format,
        data_range,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        img = self.get_rgb_image_(img, data_format, data_range)
        cv2.imwrite(filename, img)
        if name and self._wandb_logger:
            wandb.log(
                {
                    name: wandb.Image(self.get_save_path(filename)),
                    "trainer/global_step": step,
                }
            )

    def save_rgb_image(
        self,
        filename,
        img,
        data_format=DEFAULT_RGB_KWARGS["data_format"],
        data_range=DEFAULT_RGB_KWARGS["data_range"],
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        self._save_rgb_image(
            self.get_save_path(filename), img, data_format, data_range, name, step
        )

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

    def get_image_grid_(self, imgs, align):
        if isinstance(imgs[0], list):
            return np.concatenate(
                [self.get_image_grid_(row, align) for row in imgs], axis=0
            )
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

        if align == "max":
            h = max([col.shape[0] for col in cols])
            w = max([col.shape[1] for col in cols])
        elif align == "min":
            h = min([col.shape[0] for col in cols])
            w = min([col.shape[1] for col in cols])
        elif isinstance(align, int):
            h = align
            w = align
        elif (
            isinstance(align, tuple)
            and isinstance(align[0], int)
            and isinstance(align[1], int)
        ):
            h, w = align
        else:
            raise ValueError(
                f"Unsupported image grid align: {align}, should be min, max, int or (int, int)"
            )

        for i in range(len(cols)):
            if cols[i].shape[0] != h or cols[i].shape[1] != w:
                cols[i] = cv2.resize(cols[i], (w, h), interpolation=cv2.INTER_LINEAR)
        return np.concatenate(cols, axis=1)

    def save_image_grid(
        self,
        filename,
        imgs,
        align=DEFAULT_GRID_KWARGS["align"],
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
        img = self.get_image_grid_(imgs, align=align)
        filepath = self.get_save_path(filename)
        cv2.imwrite(filepath, img)
        if name and self._wandb_logger:
            wandb.log({name: wandb.Image(filepath), "trainer/global_step": step})

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

    def save_img_sequence(
        self,
        filename,
        img_dir,
        matcher,
        save_format="mp4",
        fps=30,
        name: Optional[str] = None,
        step: Optional[int] = None,
    ):
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
        if name and self._wandb_logger:
            wandb.log(
                {
                    name: wandb.Video(self.get_save_path(filename), format="mp4"),
                    "trainer/global_step": step,
                }
            )

    def save_mesh(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None):
        v_pos = self.convert_data(v_pos)
        t_pos_idx = self.convert_data(t_pos_idx)
        mesh = trimesh.Trimesh(vertices=v_pos, faces=t_pos_idx)
        mesh.export(self.get_save_path(filename))

    def save_obj(
        self,
        filename: str,
        mesh: Mesh,
        save_mat: bool = False,
        save_normal: bool = False,
        save_uv: bool = False,
        save_vertex_color: bool = False,
        map_Kd: Optional[Float[Tensor, "H W 3"]] = None,
        map_Ks: Optional[Float[Tensor, "H W 3"]] = None,
        map_Bump: Optional[Float[Tensor, "H W 3"]] = None,
        map_format: str = "jpg",
    ) -> None:
        if not filename.endswith(".obj"):
            filename += ".obj"
        v_pos, t_pos_idx = self.convert_data(mesh.v_pos), self.convert_data(
            mesh.t_pos_idx
        )
        v_nrm, v_tex, t_tex_idx, v_rgb = None, None, None, None
        if save_normal:
            v_nrm = self.convert_data(mesh.v_nrm)
        if save_uv:
            v_tex, t_tex_idx = self.convert_data(mesh.v_tex), self.convert_data(
                mesh.t_tex_idx
            )
        if save_vertex_color:
            v_rgb = self.convert_data(mesh.v_rgb)
        matname, mtllib = None, None
        if save_mat:
            matname = "default"
            mtl_filename = filename.replace(".obj", ".mtl")
            mtllib = os.path.basename(mtl_filename)
            self._save_mtl(
                mtl_filename,
                matname,
                map_Kd=self.convert_data(map_Kd),
                map_Ks=self.convert_data(map_Ks),
                map_Bump=self.convert_data(map_Bump),
                map_format=map_format,
            )
        self._save_obj(
            filename,
            v_pos,
            t_pos_idx,
            v_nrm=v_nrm,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            v_rgb=v_rgb,
            matname=matname,
            mtllib=mtllib,
        )

    def _save_obj(
        self,
        filename,
        v_pos,
        t_pos_idx,
        v_nrm=None,
        v_tex=None,
        t_tex_idx=None,
        v_rgb=None,
        matname=None,
        mtllib=None,
    ):
        obj_str = ""
        if matname is not None:
            obj_str += f"mtllib {mtllib}\n"
            obj_str += f"g object\n"
            obj_str += f"usemtl {matname}\n"
        for i in range(len(v_pos)):
            obj_str += f"v {v_pos[i][0]} {v_pos[i][1]} {v_pos[i][2]}"
            if v_rgb is not None:
                obj_str += f" {v_rgb[i][0]} {v_rgb[i][1]} {v_rgb[i][2]}"
            obj_str += "\n"
        if v_nrm is not None:
            for v in v_nrm:
                obj_str += f"vn {v[0]} {v[1]} {v[2]}\n"
        if v_tex is not None:
            for v in v_tex:
                obj_str += f"vt {v[0]} {1.0 - v[1]}\n"

        for i in range(len(t_pos_idx)):
            obj_str += "f"
            for j in range(3):
                obj_str += f" {t_pos_idx[i][j] + 1}/"
                if v_tex is not None:
                    obj_str += f"{t_tex_idx[i][j] + 1}"
                obj_str += "/"
                if v_nrm is not None:
                    obj_str += f"{t_pos_idx[i][j] + 1}"
            obj_str += "\n"

        with open(self.get_save_path(filename), "w") as f:
            f.write(obj_str)

    def _save_mtl(
        self,
        filename,
        matname,
        Ka=(0.0, 0.0, 0.0),
        Kd=(1.0, 1.0, 1.0),
        Ks=(0.0, 0.0, 0.0),
        map_Kd=None,
        map_Ks=None,
        map_Bump=None,
        map_format="jpg",
        step: Optional[int] = None,
    ):
        mtl_str = f"newmtl {matname}\n"
        mtl_str += f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
        mtl_save_path = self.get_save_path(filename)
        if map_Kd is not None:
            mtl_str += f"map_Kd texture_kd.{map_format}\n"
            self._save_rgb_image(
                os.path.join(
                    os.path.dirname(mtl_save_path), f"texture_kd.{map_format}"
                ),
                map_Kd,
                data_format="HWC",
                data_range=(0, 1),
                name=f"{matname}_Kd",
                step=step,
            )
        else:
            mtl_str += f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"
        if map_Ks is not None:
            mtl_str += f"map_Ks texture_ks.{map_format}\n"
            self._save_rgb_image(
                os.path.join(
                    os.path.dirname(mtl_save_path), f"texture_ks.{map_format}"
                ),
                map_Ks,
                data_format="HWC",
                data_range=(0, 1),
                name=f"{matname}_Ks",
                step=step,
            )
        else:
            mtl_str += f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"
        if map_Bump is not None:
            mtl_str += f"map_Bump texture_nrm.{map_format}\n"
            self._save_rgb_image(
                os.path.join(
                    os.path.dirname(mtl_save_path), f"texture_nrm.{map_format}"
                ),
                map_Bump,
                data_format="HWC",
                data_range=(0, 1),
                name=f"{matname}_Bump",
                step=step,
            )
        with open(self.get_save_path(filename), "w") as f:
            f.write(mtl_str)

    def save_file(self, filename, src_path):
        shutil.copyfile(src_path, self.get_save_path(filename))

    def save_json(self, filename, payload):
        with open(self.get_save_path(filename), "w") as f:
            f.write(json.dumps(payload))
