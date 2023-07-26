import argparse
import os

import tinycudann as tcnn
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def tcnn_encoder():
    config = {
        "otype": "DenseGrid",
        "n_levels": 5,
        "n_features_per_level": 4,
        "base_resolution": 16,
        "per_level_scale": 1.4142135623730951
    }
    in_dim = 3
    encoder = tcnn.Encoding(in_dim, config, dtype=torch.float32)
    return encoder


def random_encoder(out_dim: int = 1634048):
    encoder = nn.Parameter(torch.randn(out_dim, dtype=torch.float32).cuda(), requires_grad=True)
    return encoder


def specify_encoder(param: torch.Tensor):
    encoder = nn.Parameter(param.cuda(), requires_grad=True)
    return encoder


def load_data(path: str):
    dat = torch.from_numpy(np.load(path)).cuda()
    return dat


def relative_error(dat1: torch.Tensor, dat2: torch.Tensor):
    # eps = min(torch.minimum(torch.abs(dat1), torch.abs(dat2)).min() / 10, 1e-30)
    eps = 1e-30
    err = (torch.abs(dat1 - dat2) / (torch.abs(dat1) + torch.abs(dat2) + eps))
    return err.max(), err.mean()


def setup():
    parser = argparse.ArgumentParser("TCNN Encoder")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args


class TCNN_Emb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = tcnn_encoder()
    
    def forward(self, x, y) -> torch.Tensor:
        x = self.emb(x)
        loss = F.mse_loss(x, y)
        return loss
    

class Hyper_Emb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = tcnn_encoder()
        self.latent = nn.Parameter(torch.zeros_like(self.emb.params), requires_grad=True)
    
    def forward(self, x, y) -> torch.Tensor:
        self.emb.params = self.latent
        x = self.emb(x)
        loss = F.mse_loss(x, y)
        return loss
    

def test_module_param(mod: nn.Module) -> None:
    for k, v in mod.named_parameters():
        print(k, v.shape, v.dtype, v.requires_grad)


def main():

    setup()

    # target layer
    target_data = load_data("densegrid.npy")
    target_emb = tcnn_encoder()
    target_emb.state_dict()["params"].copy_(target_data)
    target_emb.params.requires_grad = False

    # TCNN training
    tcnn_emb = TCNN_Emb()
    tcnn_opt = optim.Adam(tcnn_emb.parameters(), lr=0.01, betas=[0.9, 0.99])
    test_module_param(tcnn_emb)

    # Hyper training
    hyper_emb = Hyper_Emb()
    hyper_opt = optim.Adam(hyper_emb.parameters(), lr=0.01, betas=[0.9, 0.99])
    test_module_param(hyper_emb)

    # err_max, err_mean = relative_error(tcnn_emb.emb.params, hyper_emb.latent)
    err_max, err_mean = relative_error(tcnn_emb.emb.params, hyper_emb.emb.params)
    print(f"Initial Relative Error {err_max:.8f}, {err_mean:.8f}")

    for i in range(10000):
        x = torch.rand(64, 3).cuda() * 2.0 - 1.0
        y = target_emb(x)

        tcnn_loss = tcnn_emb(x, y)
        tcnn_opt.zero_grad()
        tcnn_loss.backward()
        tcnn_opt.step()
        
        hyper_loss = hyper_emb(x, y)
        hyper_opt.zero_grad()
        hyper_loss.backward()
        hyper_opt.step()

        print(hyper_emb.latent.norm(), hyper_emb.latent.grad.norm())

        # if (i + 1) % 1 == 0:
        #     print(f"[{i}/10000] TCNN Loss {tcnn_loss.item():.8f} Hyper Loss {hyper_loss.item():.8f}")
            
        #     # err_max, err_mean = relative_error(tcnn_emb.emb.params, hyper_emb.latent)
        #     err_max, err_mean = relative_error(tcnn_emb.emb.params, hyper_emb.emb.params)
        #     print(f"[{i}/10000] Relative Error Param {err_max:.8f}, {err_mean:.8f}")

        #     # err_max, err_mean = relative_error(tcnn_emb.emb.params.grad, hyper_emb.latent.grad)
        #     err_max, err_mean = relative_error(tcnn_emb.emb.params.grad, hyper_emb.emb.params.grad)
        #     print(f"[{i}/10000] Relative Error Grad {err_max:.8f}, {err_mean:.8f}")

    err_max, err_mean = relative_error(tcnn_emb.emb.params, target_emb.params)
    print(f"Final Relative Error TCNN Emb {err_max:.8f}, {err_mean:.8f}")

    err_max, err_mean = relative_error(hyper_emb.latent, target_emb.params)
    print(f"Final Relative Error Hyper Emb {err_max:.8f}, {err_mean:.8f}")


main()

