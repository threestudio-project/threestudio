import os
import math
from dataclasses import dataclass

import torch
import numpy as np

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.models.geometry.gaussian import BasicPointCloud
from threestudio.utils.typing import *
from threestudio.utils.perceptual import PerceptualLoss

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

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
    matrix = np.linalg.inv(c2w[0].cpu().numpy())
    R = np.transpose(matrix[:3,:3])
    R[:,0] = -R[:,0]
    T = -matrix[:3, 3]
    
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
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
        
@threestudio.register("gaussian-splatting-editing-system")
class GaussianSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        extent: float = 5.0
        num_pts: int = 100
        invert_bg_prob: float = 0.5

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.perceptual_loss = PerceptualLoss().eval().to()
        self.automatic_optimization=False
        
        self.background_tensor = torch.tensor(
            [0, 0, 0], 
            dtype=torch.float32, 
            device="cuda"
        )
        # Since this data set has no colmap data, we start with random points
        num_pts = self.cfg.num_pts
        print(f"Generating random point cloud ({num_pts})...")
        self.extent = self.cfg.extent
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
        self.gaussians_step = 0
        
    def configure_optimizers(self):
        optim = self.geometry.optimizer
        ret = {
            "optimizer": optim,
        }
        return ret

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lr_max_step = self.geometry.cfg.position_lr_max_steps
        
        # self.geometry.update_learning_rate(self.gaussians_step)
        if self.gaussians_step < lr_max_step:
            self.geometry.update_learning_rate(self.gaussians_step)
        else:
            self.geometry.update_learning_rate_fine(self.gaussians_step - lr_max_step)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if (self.gaussians_step) >= self.opt.position_lr_max_steps:
        #     self.gaussians.oneupSHdegree()
        proj = batch['proj'][0]
        # print(proj.shape)
        # print(proj)
        fovx, fovy = getFOV(proj, 0.01, 100.0)
        # print(fovx, fovy)
        # fovy = batch['fovy'][0]
        w2c, proj, cam_p = get_cam_info(c2w=batch['c2w'], fovy=fovy, fovx=fovx)
        # print(proj)
        # exit(0)
            
        # import pdb; pdb.set_trace()
        viewpoint_cam = Camera(
            FoVx=fovy, 
            FoVy=fovy, 
            image_width=batch['width'], 
            image_height=batch['height'],
            world_view_transform=w2c,
            full_proj_transform=proj,
            camera_center=cam_p,
        )
        
        render_pkg = self.renderer(
            viewpoint_cam, 
            self.background_tensor,
        )
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # render_out = {
        #     "comp_rgb": image,
        # }
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        self.gaussians_step += 1
        out = self(batch)

        visibility_filter = out['visibility_filter']
        radii = out["radii"]
        guidance_inp = out["render"].unsqueeze(0).permute(0, 2, 3, 1)  
        # import pdb; pdb.set_trace()
        viewspace_point_tensor = out["viewspace_points"]
        
        origin_gt_rgb = batch["gt_rgb"]
        B, H, W, C = origin_gt_rgb.shape
        gt_rgb = origin_gt_rgb

        # for key in out:
        #     print(key)
        # exit(0)
        # print(gt_rgb)
        # print(out["render"])
        # import numpy as np
        # import cv2
        # show = (out["render"].detach().cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
        # cv2.imwrite("test.jpg", show)
        # exit(0)
        guidance_out = {
            "loss_l1": torch.nn.functional.l1_loss(out["render"], gt_rgb.permute(0, 3, 1, 2)[0]),
            "loss_p": self.perceptual_loss(
                out["render"].unsqueeze(0).contiguous(),
                gt_rgb.permute(0, 3, 1, 2).contiguous(),
            ).mean(),
        }

        loss = 0.0
        
        self.log("gauss_num", int(self.geometry.get_xyz.shape[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
            
            
        loss.backward()
        iteration = self.gaussians_step
        self.geometry.update_states(
            iteration,
            visibility_filter,
            radii,
            viewspace_point_tensor,
            self.extent,
        )
        opt.step()
        opt.zero_grad(set_to_none = True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.gaussians_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["render"].unsqueeze(0).permute(0, 2, 3, 1)[0],
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
                    "img": out["render"].unsqueeze(0).permute(0, 2, 3, 1)[0],
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
