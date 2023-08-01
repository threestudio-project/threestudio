import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from spec_norm import spectral_norm

import argparse
import os
from tqdm import tqdm


path = "outputs/hypernet"


def load_data(cate: str) -> torch.Tensor:
    embed = torch.from_numpy(np.load(os.path.join(path, f"embedding-{cate}.npy"))).flatten().float().cuda()
    grid = torch.from_numpy(np.load(os.path.join(path, f"densegrid-{cate}.npy"))).flatten().float().cuda()
    return embed, grid


def load_model(name: str, config: dict) -> nn.Module:
    ckpt = torch.load(os.path.join(path, f"{name}.ckpt"))
    model = HyperNet(**config).cuda()
    state_dict = model.state_dict()
    for k in state_dict.keys():
        state_dict[k] = ckpt["state_dict"]["hypernet." + k]
    model.load_state_dict(state_dict)
    return model


class HyperNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.linear1 = spectral_norm(nn.Linear(in_dim, hid_dim, bias=True))
        self.linear2 = spectral_norm(nn.Linear(hid_dim, out_dim, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.linear1(x)
            x = F.silu(x, inplace=True)
            x = self.linear2(x)
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

    x, y = load_data("bunny")

    config = {
        "in_dim": x.shape[0],
        "hid_dim": 32,
        "out_dim": y.shape[0]
    }
    print(f"X Shape {x.shape}, Y Shape {y.shape}")

    net = HyperNet(**config).cuda()
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


def test_hyper_net():

    setup()

    embed_bunny, grid_bunny = load_data("bunny")
    embed_pig, grid_pig = load_data("pig")

    config = {
        "in_dim": embed_bunny.shape[0],
        "hid_dim": 32,
        "out_dim": grid_bunny.shape[0]
    }

    model = load_model("model", config)

    print(f"Embedding Error {F.mse_loss(embed_bunny, embed_pig)}")
    print(f"Densegrid Error {F.mse_loss(grid_bunny, grid_pig)}")
    print(f"Model Output Error {model(embed_bunny, grid_bunny)}")
    print(f"Model Output Error {model(embed_pig, grid_pig)}")


test_hyper_net()
