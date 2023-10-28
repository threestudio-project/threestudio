import math
import os
from dataclasses import dataclass

import numpy as np
import torch

import threestudio
from threestudio.models.geometry.gaussian import BasicPointCloud
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *


def convert_pose(C2W):
    flip_yz = np.eye(4)
    # flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getFOV(P, znear, zfar):
    right = znear / P[0, 0]
    top = znear / P[1, 1]
    tanHalfFovX = right / znear
    tanHalfFovY = top / znear
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info(c2w, fovx, fovy):
    c2w = c2w[0].cpu().numpy()
    c2w = convert_pose(c2w)
    world_view_transform = np.linalg.inv(c2w)

    world_view_transform = (
        torch.tensor(world_view_transform).transpose(0, 1).cuda().float()
    )
    projection_matrix = (
        getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)
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


@threestudio.register("dynamic-gaussian-splatting-instruct-system")
class DynamicGaussianSplattingInstruct(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        extent: float = 5.0
        num_pts: int = 100
        invert_bg_prob: float = 0.5

        per_editing_step: int = 10
        start_editing_step: int = 1000

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.perceptual_loss = PerceptualLoss().eval().to()
        self.automatic_optimization = False

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        # Since this data set has no colmap data, we start with random points
        num_pts = self.cfg.num_pts

        self.extent = self.cfg.extent

        if len(self.geometry.cfg.geometry_convert_from) == 0:
            print(f"Generating random point cloud ({num_pts})...")
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = 0.25 * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)

            shs = np.random.random((num_pts, 3)) / 255.0
            C0 = 0.28209479177387814
            color = shs * C0 + 0.5
            pcd = BasicPointCloud(
                points=xyz, colors=color, normals=np.zeros((num_pts, 3))
            )

            self.geometry.create_from_pcd(pcd, 10)
            self.geometry.training_setup()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        self.edit_frames = {}

    def configure_optimizers(self):
        g_optim = self.geometry.gaussian_optimizer
        d_optim = parse_optimizer(self.cfg.optimizer, self)

        return g_optim, d_optim

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.gaussian.cfg.position_lr_max_steps

        if self.global_step // 2 < lr_max_step:
            self.geometry.gaussian.update_learning_rate(self.global_step // 2)
        else:
            self.geometry.gaussian.update_learning_rate_fine(
                self.global_step // 2 - lr_max_step
            )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if (self.gaussians_step) >= self.opt.position_lr_max_steps:
        #     self.gaussians.oneupSHdegree()
        proj = batch["proj"][0]
        fovx, fovy = getFOV(proj, 0.01, 100.0)
        w2c, proj, cam_p = get_cam_info(c2w=batch["c2w"], fovy=fovy, fovx=fovx)

        viewpoint_cam = Camera(
            FoVx=fovx,
            FoVy=fovy,
            image_width=batch["width"],
            image_height=batch["height"],
            world_view_transform=w2c,
            full_proj_transform=proj,
            camera_center=cam_p,
        )

        render_pkg = self.renderer(
            viewpoint_cam,
            batch["moment"][0],
            self.background_tensor,
        )

        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def on_load_checkpoint(self, ckpt_dict) -> None:
        num_pts = ckpt_dict["state_dict"]["geometry.gaussian._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()
        super().on_load_checkpoint(ckpt_dict)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        origin_gt_rgb = batch["gt_rgb"]
        B, H, W, C = origin_gt_rgb.shape
        if batch_index in self.edit_frames:
            gt_rgb = self.edit_frames[batch_index].to(batch["gt_rgb"].device)
            gt_rgb = torch.nn.functional.interpolate(
                gt_rgb.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            batch["gt_rgb"] = gt_rgb
        else:
            gt_rgb = origin_gt_rgb

        if (
            self.cfg.per_editing_step > 0
            and self.global_step > self.cfg.start_editing_step
        ):
            prompt_utils = self.prompt_processor()
            if (
                not batch_index in self.edit_frames
                or self.global_step % self.cfg.per_editing_step == 0
            ):
                self.renderer.eval()
                full_out = self(batch)
                self.renderer.train()
                result = self.guidance(
                    full_out["refine"].unsqueeze(0).permute(0, 2, 3, 1),
                    origin_gt_rgb,
                    prompt_utils,
                )
                self.edit_frames[batch_index] = result["edit_images"].detach().cpu()

        out = self(batch)

        visibility_filter = out["visibility_filter"]
        radii = out["radii"]
        guidance_inp = out["render"].unsqueeze(0).permute(0, 2, 3, 1)
        viewspace_point_tensor = out["viewspace_points"]

        bg_color = out["bg_color"]
        # mask = torch.sum(origin_gt_rgb, dim=-1).unsqueeze(-1)
        # mask = (mask > 1e-3).float()
        # gt_rgb = gt_rgb * mask + (1 - mask) * bg_color.reshape(1, 1, 1, 3)

        guidance_out = {
            "loss_l1": torch.nn.functional.l1_loss(
                out["render"], gt_rgb.permute(0, 3, 1, 2)[0]
            ),
            "loss_p": self.perceptual_loss(
                out["render"].unsqueeze(0).contiguous(),
                gt_rgb.permute(0, 3, 1, 2).contiguous(),
            ).mean(),
            "loss_G_l1": torch.nn.functional.l1_loss(
                out["refine"], gt_rgb.permute(0, 3, 1, 2)[0]
            ),
            "loss_G_p": self.perceptual_loss(
                out["refine"].unsqueeze(0).contiguous(),
                gt_rgb.permute(0, 3, 1, 2).contiguous(),
            ).mean(),
            "loss_xyz_residual": torch.mean(self.geometry.gaussian._xyz_residual**2),
            "loss_scaling_residual": torch.mean(
                self.geometry.gaussian._xyz_residual**2
            ),
        }

        loss = 0.0
        loss_l1 = guidance_out["loss_l1"] * self.C(self.cfg.loss["lambda_l1"])

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
            if name.startswith("loss_") and (not name.startswith("loss_l1")):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        loss_l1.backward(retain_graph=True)
        iteration = self.global_step // 2
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
            self.extent,
        )
        loss.backward()
        g_opt.step()
        d_opt.step()
        g_opt.zero_grad(set_to_none=True)
        d_opt.zero_grad(set_to_none=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        if batch_index in self.edit_frames:
            B, H, W, C = batch["gt_rgb"].shape
            rgb = torch.nn.functional.interpolate(
                self.edit_frames[batch_index].permute(0, 3, 1, 2), (H, W)
            ).permute(0, 2, 3, 1)[0]
        else:
            rgb = batch["gt_rgb"][0]
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.jpg",
            [
                {
                    "type": "rgb",
                    "img": out["render"].unsqueeze(0).permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["refine"].unsqueeze(0).permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": rgb,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
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
            f"it{self.global_step}-test/{batch['index'][0]}.jpg",
            [
                {
                    "type": "rgb",
                    "img": out["render"].unsqueeze(0).permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["refine"].unsqueeze(0).permute(0, 2, 3, 1)[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ],
            name="test_step",
            step=self.global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            "(\d+)\.jpg",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.global_step,
        )
