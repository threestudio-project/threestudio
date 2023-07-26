import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from spec_norm import spectral_norm

import argparse
import os
from tqdm import tqdm


dg = "outputs/hypernet/densegrid.npy"
pr = "outputs/hypernet/prompt.npy"


def load_data(path: str) -> torch.Tensor:
    dat = torch.from_numpy(np.load(path)).flatten().float().cuda()
    return dat


class Hyper(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.lin1 = spectral_norm(nn.Linear(in_dim, hid_dim, bias=True))
        self.lin2 = spectral_norm(nn.Linear(hid_dim, out_dim, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.lin1(x)
            x = F.silu(x, inplace=True)
            x = self.lin2(x)
            loss = F.mse_loss(x, y)
        return loss
    

def setup():
    parser = argparse.ArgumentParser("TCNN Encoder")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args
    

def train_hyper_net():

    setup()

    x = load_data(pr)
    y = load_data(dg)

    config = {
        "in_dim": x.shape[0],
        "hid_dim": 32,
        "out_dim": y.shape[0]
    }

    print(f"X Shape {x.shape}, Y Shape {y.shape}")

    net = Hyper(**config).cuda()
    opt = optim.Adam(net.parameters(), lr=0.1, betas=[0.9, 0.999])

    with tqdm(range(10000)) as loader:
        for _ in loader:
            loss = net(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loader.set_postfix(
                ordered_dict={
                    "Loss": loss.item()
                }
            )


train_hyper_net()
