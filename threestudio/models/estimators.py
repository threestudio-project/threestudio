from typing import Callable, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from nerfacc.data_specs import RayIntervals
from nerfacc.estimators.base import AbstractEstimator
from nerfacc.pdf import importance_sampling, searchsorted
from nerfacc.volrend import render_transmittance_from_density
from torch import Tensor


class ImportanceEstimator(AbstractEstimator):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @torch.no_grad()
    def sampling(
        self,
        prop_sigma_fns: List[Callable],
        prop_samples: List[int],
        num_samples: int,
        # rendering options
        n_rays: int,
        near_plane: float,
        far_plane: float,
        sampling_type: Literal["uniform", "lindisp"] = "uniform",
        # training options
        stratified: bool = False,
        requires_grad: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sampling with CDFs from proposal networks.

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), (
            "The number of proposal networks and the number of samples "
            "should be the same."
        )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        for level_fn, level_samples in zip(prop_sigma_fns, prop_samples):
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, stratified
            )
            t_vals = _transform_stot(
                sampling_type, intervals.vals, near_plane, far_plane
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas)
                cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)

        intervals, _ = importance_sampling(intervals, cdfs, num_samples, stratified)
        t_vals_fine = _transform_stot(
            sampling_type, intervals.vals, near_plane, far_plane
        )

        t_vals = torch.cat([t_vals, t_vals_fine], dim=-1)
        t_vals, _ = torch.sort(t_vals, dim=-1)

        t_starts_ = t_vals[..., :-1]
        t_ends_ = t_vals[..., 1:]

        return t_starts_, t_ends_


def _transform_stot(
    transform_type: Literal["uniform", "lindisp"],
    s_vals: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
) -> torch.Tensor:
    if transform_type == "uniform":
        _contract_fn, _icontract_fn = lambda x: x, lambda x: x
    elif transform_type == "lindisp":
        _contract_fn, _icontract_fn = lambda x: 1 / x, lambda x: 1 / x
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)
