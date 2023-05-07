import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import threestudio


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        threestudio.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adan"]:
        from threestudio.systems import optimizers

        optim = getattr(optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
