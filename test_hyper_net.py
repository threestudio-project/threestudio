import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from spec_norm import spectral_norm

from tqdm import tqdm, trange


df = "./outputs/dreamfusion-if-lowres/a_pig_wearing_medieval_armor_holding_a_blue_balloon@20230717-154644/ckpts/epoch=0-step=10000.ckpt"
dg = "./outputs/att3d-if/a_pig_wearing_medieval_armor_holding_a_blue_balloon@20230717-145300/save/densegrid.npy"
pr = "./outputs/dreamfusion-if-lowres/a_pig_wearing_medieval_armor_holding_a_blue_balloon@20230717-170246/save/prompt.npy"


def load_data(path: str) -> torch.Tensor:
    if path.endswith(".ckpt"):
        dat = torch.load(path)["state_dict"]["geometry.encoding.encoding.encoding.params"]
    else:
        dat = torch.from_numpy(np.load(path))
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
    

def train_hyper_net():

    x = load_data(pr).flatten().float().cuda()
    y = load_data(df).cuda()

    print(f"[DATA] X: {x.shape} {x.dtype}, Y: {y.shape} {y.dtype}")

    config = {
        "in_dim": x.shape[0],
        "hid_dim": 32,
        "out_dim": y.shape[0]
    }

    net = Hyper(**config).cuda()
    opt = optim.Adam(net.parameters(), lr=1e-4, betas=[0.9, 0.999])

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
