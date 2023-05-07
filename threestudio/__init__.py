__modules__ = {}


def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


###  grammar sugar for logging utilities  ###
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_warn,
)

debug = rank_zero_debug
info = rank_zero_info
warn = rank_zero_warn


from . import data, models, systems
