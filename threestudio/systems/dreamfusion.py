from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.typing import *
from threestudio.utils.ops import dot, binary_cross_entropy

@threestudio.register("dreamfusion-system")
class DreamFusion(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        geometry_type: str = "implicit-volume"
        geometry: dict = field(default_factory=dict)
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

    cfg: Config

    def configure(self):
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(self.cfg.background)
        self.renderer = threestudio.find(self.cfg.renderer_type)(self.cfg.renderer, geometry=self.geometry, material=self.material, background=self.background)
        # self.automatic_optimization = False

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
        # opt = self.optimizers()
        # opt.zero_grad()

        out = self(batch)
        text_embeddings = self.prompt_processor(**batch)
        guidance_out = self.guidance(out['comp_rgb'], text_embeddings, rgb_as_latents=False) 

        loss = 0.

        loss += guidance_out['sds'] * self.C(self.cfg.loss.lambda_sds)

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if 'normal' not in out:
                raise ValueError("Normal is required for orientation loss, no normal is found in the output.")
            loss_orient = (out['weights'].detach() * dot(out['normal'], out['t_dirs']).clamp_min(0.)**2).sum() / (out['opacity'] > 0).sum()
            self.log('train/loss_orient', loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.cfg.loss.lambda_distortion) > 0:
            from torch_efficient_distloss import flatten_eff_distloss        
            loss_distortion = flatten_eff_distloss(out['weights'][...,0], out['t_points'][...,0], out['t_intervals'][...,0], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.cfg.loss.lambda_distortion)            

        loss_sparsity = (out['opacity']**2 + 0.01).sqrt().mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out['opacity'].clamp(1.e-3, 1.-1.e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f'train_params/{name}', self.C(value))

        return {
            'loss': loss
        }
        # self.manual_backward(loss)
        # opt.step()
        # sch = self.lr_schedulers()
        # sch.step()

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
        return
        out = self(batch)
        self.save_image_grid(f"it{self.global_step}-test/{batch_idx}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'][0], 'kwargs': {'data_format': 'HWC'}},
        ] + ([
            {'type': 'rgb', 'img': out['comp_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (0, 1)}}
        ] if 'comp_normal' in out else []) + [
            {'type': 'grayscale', 'img': out['opacity'][0,:,:,0], 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ])

    def on_test_epoch_end(self):
        # import pdb; pdb.set_trace();
        mesh = self.geometry.isosurface()
        self.save_mesh(f"mesh.obj", mesh.v_pos, mesh.t_pos_idx)
        return
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