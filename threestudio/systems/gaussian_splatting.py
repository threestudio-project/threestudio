import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.cuda.amp import autocast

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *


def stack_dicts(dicts):
    stacked_dict = {}
    for k in dicts[0].keys():
        stacked_dict[k] = torch.stack([d[k] for d in dicts], dim=0)
    return stacked_dict


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getFOV(P):
    tanHalfFovX = 1 / P[0, 0]
    tanHalfFovY = 1 / P[1, 1]
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info(c2w, fovx, fovy, znear, zfar):
    c2w = c2w.cpu().numpy()
    c2w = convert_pose(c2w)
    world_view_transform = np.linalg.inv(c2w)

    world_view_transform = (
        torch.tensor(world_view_transform).transpose(0, 1).cuda().float()
    )
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        .transpose(0, 1)
        .cuda()
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


class Camera(NamedTuple):
    FoVx: torch.Tensor
    FoVy: torch.Tensor
    camera_center: torch.Tensor
    image_width: int
    image_height: int
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor


@threestudio.register("gaussian-splatting-system")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        invert_bg_prob: float = 0.5

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.automatic_optimization = False

        self.background_tensor = torch.tensor(
            [1, 1, 1], dtype=torch.float32, device="cuda"
        )

        self.geometry.init_params()
        self.geometry.training_setup()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.gaussians_step = 0

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        ret = {
            "optimizer": optim,
        }
        return ret

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.cfg.position_lr_max_steps
        scale_lr_max_steps = self.geometry.cfg.scale_lr_max_steps

        # self.geometry.update_learning_rate(self.gaussians_step)
        if self.gaussians_step < lr_max_step:
            self.geometry.update_xyz_learning_rate(self.gaussians_step)

        if self.gaussians_step < scale_lr_max_steps:
            self.geometry.update_scale_learning_rate(self.gaussians_step)

        bs = batch["c2w"].shape[0]
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []
        for batch_idx in range(bs):
            fovy = batch["fovy"][batch_idx]
            w2c, proj, cam_p = get_cam_info(
                c2w=batch["c2w"][batch_idx], fovx=fovy, fovy=fovy, znear=0.1, zfar=100
            )

            # import pdb; pdb.set_trace()
            viewpoint_cam = Camera(
                FoVx=fovy,
                FoVy=fovy,
                image_width=batch["width"],
                image_height=batch["height"],
                world_view_transform=w2c,
                full_proj_transform=proj,
                camera_center=cam_p,
            )

            with autocast(enabled=False):
                render_pkg = self.renderer(
                    viewpoint_cam,
                    self.background_tensor,
                )
                renders.append(render_pkg["render"])
                viewspace_points.append(render_pkg["viewspace_points"])
                visibility_filters.append(render_pkg["visibility_filter"])
                radiis.append(render_pkg["radii"])
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # render_out = {
        #     "comp_rgb": image,
        # }
        outputs = {
            "render": torch.stack(renders, dim=0),
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filters,
            "radii": radiis,
        }
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        self.gaussians_step += 1
        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["render"].permute(0, 2, 3, 1)
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        guidance_out = self.guidance(
            guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
        )

        loss_sds = 0.0
        loss = 0.0

        self.log(
            "gauss_num",
            int(self.geometry.get_xyz.shape[0]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss_sds += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        xyz_mean = None
        if self.cfg.loss["lambda_position"] > 0.0:
            xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            self.log(f"train/position", value)
            loss += self.C(self.cfg.loss["lambda_position"]) * xyz_mean.mean()

        if self.cfg.loss["lambda_opacity"] > 0.0:
            if xyz_mean is None:
                xyz_mean = self.geometry.get_xyz.norm(dim=-1)
            self.log(f"train/opacity", value)
            loss += (
                self.C(self.cfg.loss["lambda_opacity"])
                * (xyz_mean.detach() * self.geometry.get_opacity).mean()
            )

        if self.cfg.loss["lambda_scales"] > 0.0:
            scale_mean = torch.mean(self.geometry.get_scaling)
            self.log(f"train/scales", scale_mean)
            loss += self.C(self.cfg.loss["lambda_scales"]) * scale_mean

        def _tensor_size(t):
            return t.size()[1] * t.size()[2] * t.size()[3]

        def tv_loss(x):
            batch_size = x.size()[0]
            h_x = x.size()[2]
            w_x = x.size()[3]
            count_h = _tensor_size(x[:, :, 1:, :])
            count_w = _tensor_size(x[:, :, :, 1:])
            h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
            w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
            return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        loss_tv = tv_loss(out["render"]) * 0.0
        self.log(f"train/loss_tv", loss_tv)
        loss += loss_tv

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_sds.backward(retain_graph=True)
        iteration = self.gaussians_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
        )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.gaussians_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["render"].permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.gaussians_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["render"].permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="test_step",
            step=self.global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.gaussians_step}-test",
            f"it{self.gaussians_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
