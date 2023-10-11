from dataclasses import dataclass

import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseGeometry
from threestudio.models.renderers.base import Rasterizer
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *

import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

@threestudio.register("diff-gaussian-rasterizer")
class DiffGaussian(Rasterizer):
    
    @dataclass
    class Config(Rasterizer.Config):
        debug: bool = False
        invert_bg_prob: bool = True
        
    cfg: Config
    
    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        
    def forward(
        self, 
        viewpoint_camera, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None
    ) -> Dict[str, Any]:
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        
        if self.training:
            invert_bg_color = np.random.rand() > self.cfg.invert_bg_prob
        else:
            invert_bg_color = True
            
        bg_color = bg_color if not invert_bg_color else (1.0 - bg_color)
    
        pc = self.geometry
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        # Retain gradients of the 2D (screen-space) means for batch dim
        if self.training:
            screenspace_points.retain_grad()

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}