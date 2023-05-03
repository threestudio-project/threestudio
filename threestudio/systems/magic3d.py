import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.typing import *
from threestudio.utils.ops import dot, binary_cross_entropy
from threestudio.utils.misc import cleanup, get_device


@threestudio.register("magic3d-system")
class Magic3D(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-volume"
        geometry: dict = field(default_factory=dict)

        # only used when refinement=True and from_coarse=True
        geometry_coarse_type: str = "implicit-volume"
        geometry_coarse: dict = field(default_factory=dict)

        material_type: str = "diffuse-with-point-light-material"
        material: dict = field(default_factory=dict)

        background_type: str = "neural-environment-map-background"
        background: dict = field(default_factory=dict)

        renderer_type: str = "nerf-volume-renderer"
        renderer: dict = field(default_factory=dict)

        guidance_type: str = "stable-diffusion-guidance"
        guidance: dict = field(default_factory=dict)

        prompt_processor_type: str = "dreamfusion-prompt-processor"
        prompt_processor: dict = field(default_factory=dict)

        refinement: bool = False
        from_coarse: bool = False
        inherit_coarse_texture: bool = True

    cfg: Config

    def configure(self) -> None:
        if self.cfg.refinement and self.cfg.from_coarse:
            # load coarse stage geometry
            assert self.cfg.weights is not None, "weights must be specified to initilize from coarse stage model"
            from threestudio.utils.config import load_config, parse_structured
            coarse_cfg = load_config(os.path.join(os.path.dirname(self.cfg.weights), '../configs/parsed.yaml')) # TODO: hard-coded relative path
            coarse_system_cfg: Magic3D.Config = parse_structured(self.Config, coarse_cfg.system)
            self.geometry = threestudio.find(coarse_system_cfg.geometry_type)(coarse_system_cfg.geometry)
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        if self.cfg.refinement:
            self.background.requires_grad_(False)
    
    def post_configure(self) -> None:
        # this hook is triggered after loading weights
        if self.cfg.from_coarse:
            # convert from coarse stage geometry
            assert self.cfg.refinement, "from_coarse is only valid when refinement is enabled"
            self.geometry = self.geometry.to(get_device())
            geometry_refine = threestudio.find(self.cfg.geometry_type).create_from(self.geometry, self.cfg.geometry, copy_net=self.cfg.inherit_coarse_texture)
            del self.geometry
            cleanup()
            self.geometry = geometry_refine
        
        # FIXME: renderer should not have any weights to be loaded
        self.renderer = threestudio.find(self.cfg.renderer_type)(self.cfg.renderer, geometry=self.geometry, material=self.material, background=self.background)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }
    
    def on_fit_start(self) -> None:
        """
        Initialize guidance and prompt processor in this hook:
        (1) excluded from optimizer parameters (this hook executes after optimizer is initialized)
        (2) only used in training
        To avoid being saved to checkpoints, see on_save_checkpoint below.
        """
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)
        guidance_out = self.guidance(out['comp_rgb'], text_embeddings, rgb_as_latents=False) 

        loss = 0.

        loss += guidance_out['sds'] * self.C(self.cfg.loss.lambda_sds)

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if 'normal' not in out:
                    raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
                loss_orient = (out['weights'].detach() * dot(out['normal'], out['t_dirs']).clamp_min(0.)**2).sum() / (out['opacity'] > 0).sum()
                self.log('train/loss_orient', loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)         

            loss_sparsity = (out['opacity']**2 + 0.01).sqrt().mean()
            self.log('train/loss_sparsity', loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out['opacity'].clamp(1.e-3, 1.-1.e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log('train/loss_opaque', loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        else:
            loss_normal_consistency = out['mesh'].normal_consistency()
            self.log('train/loss_normal_consistency', loss_normal_consistency)
            loss += loss_normal_consistency * self.C(self.cfg.loss.lambda_normal_consistency)

        for name, value in self.cfg.loss.items():
            self.log(f'train_params/{name}', self.C(value))

        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(f"it{self.global_step}-{batch_idx}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'][0], 'kwargs': {'data_format': 'HWC'}},
        ] + ([
            {'type': 'rgb', 'img': out['comp_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}}
        ] if 'comp_normal' in out else []) + [
            {'type': 'grayscale', 'img': out['opacity'][0,:,:,0], 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ])
    
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(f"it{self.global_step}-test/{batch_idx}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'][0], 'kwargs': {'data_format': 'HWC'}},
        ] + ([
            {'type': 'rgb', 'img': out['comp_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}}
        ] if 'comp_normal' in out else []) + [
            {'type': 'grayscale', 'img': out['opacity'][0,:,:,0], 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ])

    def on_test_epoch_end(self):
        mesh = self.geometry.isosurface()
        self.save_mesh(f"mesh.obj", mesh.v_pos, mesh.t_pos_idx)        
        self.save_img_sequence(
            f"it{self.global_step}-test",
            f"it{self.global_step}-test",
            '(\d+)\.png',
            save_format='mp4',
            fps=30
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove stable diffusion weights
        # TODO: better way?
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if k.split('.')[0] not in ['prompt_processor', 'guidance']}
        return super().on_save_checkpoint(checkpoint)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # debug use
        pass
        # from lightning.pytorch.utilities import grad_norm
        # norms = grad_norm(self.geometry, norm_type=2)
        # print(norms)