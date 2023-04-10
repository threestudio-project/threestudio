import os
import re
from packaging import version

import torch

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
    return torch.device(f'cuda:{get_rank()}')

def load_module_weights(path, module_name=None, ignore_modules=None, map_location=None) -> dict:
    if module_name is not None and ignore_modules is not None:
        raise ValueError('module_name and ignore_modules cannot be both set')
    if map_location is None:
        map_location = get_device()
    state_dict = torch.load(path, map_location=map_location)['state_dict']
        
    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any([k.startswith(ignore_module + '.') for ignore_module in ignore_modules])
            if ignore:
                continue
            state_dict_to_load[k] = v
        return state_dict_to_load

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf'^{module_name}\.(.*)$', k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v
        return state_dict_to_load

    return state_dict
