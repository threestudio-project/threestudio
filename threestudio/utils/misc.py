import gc
import math
import os
import re

import tinycudann as tcnn
import torch
from packaging import version

from threestudio.utils.config import config_to_primitive
from threestudio.utils.typing import *


def parse_version(ver: str):
    return version.parse(ver)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        map_location = get_device()

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, epoch: int, global_step: int, interpolation="linear") -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        if len(value) >= 6:
            select_i = 3
            for i in range(3, len(value) - 2, 2):
                if global_step >= value[i]:
                    select_i = i + 2
            if select_i != 3:
                start_value, start_step = value[select_i - 3], value[select_i - 2]
            else:
                start_step, start_value = value[:2]
            end_value, end_step = value[select_i - 1], value[select_i]
            value = [start_step, start_value, end_value, end_step]
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
        elif isinstance(end_step, float):
            current_step = epoch
        t = max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        if interpolation == "linear":
            value = start_value + (end_value - start_value) * t
        elif interpolation == "exp":
            value = math.exp(math.log(start_value) * (1 - t) + math.log(end_value) * t)
        else:
            raise ValueError(
                f"Unknown interpolation method: {interpolation}, only support linear and exp"
            )
    return value


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def finish_with_cleanup(func: Callable):
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        cleanup()
        return out

    return wrapper


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def enable_gradient(model, enabled: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


def find_last_path(path: str):
    if (path is not None) and ("LAST" in path):
        path = path.replace(" ", "_")
        base_dir_prefix, suffix = path.split("LAST", 1)
        base_dir = os.path.dirname(base_dir_prefix)
        prefix = os.path.split(base_dir_prefix)[-1]
        base_dir_prefix = os.path.join(base_dir, prefix)
        all_path = os.listdir(base_dir)
        all_path = [os.path.join(base_dir, dir) for dir in all_path]
        filtered_path = [dir for dir in all_path if dir.startswith(base_dir_prefix)]
        filtered_path.sort(reverse=True)
        last_path = filtered_path[0]
        new_path = last_path + suffix
        if os.path.exists(new_path):
            return new_path
        else:
            raise FileNotFoundError(new_path)
    else:
        return path
