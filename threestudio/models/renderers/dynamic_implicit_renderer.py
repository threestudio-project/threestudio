from dataclasses import dataclass, field
from functools import partial

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *


@threestudio.register("dynamic-implicit-renderer")
class DynamicImplicitRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        implicit_renderer: str = ""
        implicit_renderer_config: Optional[VolumeRenderer.Config] = None

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        self.implicit_renderer = threestudio.find(self.cfg.implicit_renderer)(
            self.cfg.implicit_renderer_config,
            geometry=geometry,
            material=material,
            background=background,
        )
        self.geometry = geometry

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        assert kwargs["moment"] is not None
        self.geometry.moment = kwargs["moment"]
        return self.implicit_renderer(rays_o, rays_d, light_positions, bg_color, **kwargs)

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        pass
        # if self.cfg.estimator == "occgrid":
        #     if self.cfg.grid_prune:

        #         def occ_eval_fn(x):
        #             density = self.geometry.forward_density(x)
        #             # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
        #             return density * self.render_step_size

        #         if self.training and not on_load_weights:
        #             self.estimator.update_every_n_steps(
        #                 step=global_step, occ_eval_fn=occ_eval_fn
        #             )
        # elif self.cfg.estimator == "proposal":
        #     if self.training:
        #         requires_grad = self.proposal_requires_grad_fn(global_step)
        #         self.vars_in_forward["requires_grad"] = requires_grad
        #     else:
        #         self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        pass
        # if self.cfg.estimator == "proposal" and self.training:
        #     self.estimator.update_every_n_steps(
        #         self.vars_in_forward["trans"],
        #         self.vars_in_forward["requires_grad"],
        #         loss_scaler=1.0,
        #     )

    def train(self, mode=True):
        return self.implicit_renderer.train(mode)

    def eval(self):
        return self.implicit_renderer.eval()
