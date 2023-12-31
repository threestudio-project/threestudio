__modules__ = {}
__version__ = "0.2.2"


def register(name):
    def decorator(cls):
        if name in __modules__:
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
        return cls

    return decorator


def find(name):
    if ":" in name:
        main_name, sub_name = name.split(":")
        NewClass = type(
            f"{main_name}.{sub_name}",
            (__modules__[main_name], __modules__[sub_name]),
            {},
        )
        return NewClass
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, systems
