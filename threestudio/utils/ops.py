import gc
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn

from threestudio.utils.typing import *


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]

def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert B is not None, "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return

    out_merged = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


def get_ray_directions(H: int, W: int, focal: float) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + 0.5,
        torch.arange(H, dtype=torch.float32) + 0.5,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"], c2w: Float[Tensor, "... 3 4"], keepdim=False
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)
    elif directions.ndim == 4: # (B, H, W, 3)
        assert c2w.ndim == 3 # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1) # (B, H, W, 3)
        rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(fovy / 2.0) # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(c2w: Float[Tensor, "B 3 4"], proj_mtx: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1        
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:,:3,:3] = c2w[:,:3,:3].permute(0,2,1)
    w2c[:,:3,3:] = -c2w[:,:3,:3].permute(0,2,1) @ c2w[:,:3,3:]
    w2c[:,3,3] = 1.
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx   


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()
