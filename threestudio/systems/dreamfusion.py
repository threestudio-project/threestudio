from dataclasses import dataclass, field
import threestudio
from threestudio.systems.base import BaseSystem
from threestudio.utils.typing import *
from threestudio.utils.ops import dot

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

        loss_sds = guidance_out['sds']
        self.log('train/loss_sds', loss_sds)
        loss += loss_sds * self.C(self.cfg.loss.lambda_sds)

        loss_orient = (out['weights'].detach() * dot(out['normal'], out['t_dirs']).clamp_min(0.)**2).sum() / (out['opacity'] > 0).sum()
        self.log('train/loss_orient', loss_orient)
        loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_opacity = (out['opacity']**2 + 0.01).sqrt().mean()
        self.log('train/loss_opacity', loss_opacity)
        loss += loss_opacity * self.C(self.cfg.loss.lambda_opacity)

        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(f"it{self.global_step}-{batch_idx}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'][0], 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'rgb', 'img': out['comp_pred_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'grayscale', 'img': out['opacity'][0,:,:,0], 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ])
    
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(f"it{self.global_step}-test/{batch_idx}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'][0], 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'rgb', 'img': out['comp_pred_normal'][0], 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'grayscale', 'img': out['opacity'][0,:,:,0], 'kwargs': {'cmap': None, 'data_range': (0, 1)}},
        ])

    def on_test_epoch_end(self):
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
