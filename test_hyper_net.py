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


class HyperNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, batch_size: int, calc_type: bool, **kwargs) -> None:
        super().__init__()
        self.linear1 = spectral_norm(self.make_linear(in_dim, hid_dim, bias=True))
        self.linear2 = spectral_norm(self.make_linear(hid_dim, out_dim, bias=False, init_range=1e-3))
        self.calc_type = calc_type
        self.param = nn.parameter.Parameter(torch.randn(batch_size, hid_dim).cuda(), requires_grad=True)

    def make_linear(self, in_dim, out_dim, bias, init_range=None):
        layer = nn.Linear(in_dim, out_dim, bias=bias)
        if init_range is not None:
            nn.init.uniform_(layer.weight, a=-init_range, b=init_range)
            if bias:
                nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.cuda.amp.autocast(enabled=False):
            if self.calc_type:
                x = self.linear1(x)
                mid = F.silu(x, inplace=True)
                x = self.linear2(mid)
                loss = F.mse_loss(x, y)
            else:
                x = self.linear2(self.param)
                mid = self.param
                loss = F.mse_loss(x, y)

        return {
            "loss": loss,
            "mid": mid,
            "pred": x
        }


def load_embed(cate: str) -> torch.Tensor:
    dat = torch.from_numpy(
        np.load(os.path.join(path, f"embedding-{cate}.npy"))).flatten().float().cuda()
    return dat


def load_grid(cate: str) -> torch.Tensor:
    dat = torch.from_numpy(
        np.load(os.path.join(path, f"densegrid-{cate}.npy"))).flatten().float().cuda()
    return dat


def load_model(ckpt_name: str, config: dict) -> HyperNet:
    ckpt = torch.load(os.path.join(path, f"{ckpt_name}.ckpt"))["state_dict"]
    model = HyperNet(**config).cuda()
    state_dict = model.state_dict()
    for k in state_dict.keys():
        state_dict[k] = ckpt["hypernet." + k]
    model.load_state_dict(state_dict)
    return model


def save_model(save_name: str, model: HyperNet) -> None:
    ckpt_name = "last"
    ckpt = torch.load(os.path.join(path, f"{ckpt_name}.ckpt"))
    state_dict = model.state_dict()
    for k in state_dict.keys():
        ckpt["state_dict"]["hypernet." + k] = state_dict[k]
    
    torch.save(ckpt, os.path.join(path, f"{save_name}.ckpt"))


def save_model_grid(ckpt_name: str, save_name: str, grid: torch.Tensor) -> None:
    ckpt = torch.load(os.path.join(path, f"{ckpt_name}.ckpt"))
    ckpt["state_dict"]["geometry.encoding.encoding.encoding.params"] = grid
    torch.save(ckpt, os.path.join(path, f"{save_name}.ckpt"))


def setup():
    parser = argparse.ArgumentParser("TCNN Encoder")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--type", default=True, type=bool)
    parser.add_argument("--dim", default=6, type=int)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args


def train_hyper_net():

    args = setup()

    # cates = ["hamburger"]
    cates = ["pineapple", "hamburger"]
    xs = torch.stack([load_embed(cate) for cate in cates])
    ys = torch.stack([load_grid(cate) for cate in cates])

    config = {
        "in_dim": xs.shape[1],
        "hid_dim": args.dim,
        "out_dim": ys.shape[1],
        "batch_size": xs.shape[0],
        "calc_type": args.type
    }
    print(f"X Shape {xs.shape}, Y Shape {ys.shape}")
    print(f"X Range ({xs[0].min()}, {xs[0].max()}), Mean {xs[0].mean()}")
    print(f"Y Range ({ys[0].min()}, {ys[0].max()}), Mean {ys[0].mean()} Norm {ys[0].norm()}")

    net = HyperNet(**config).cuda()
    opt = optim.Adam(net.parameters(), lr=0.1, betas=[0.9, 0.999])
    for k, v in net.named_parameters():
        print(k, v.shape, v.requires_grad)

    with tqdm(range(10000)) as loader:
        for ind in loader:
            out = net(xs, ys)
            print(f"[{ind}/1000] Mid {out['mid']}")
            print(f"[{ind}/1000] Out Range ({out['pred'].min()}, {out['pred'].max()})")

            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            loader.set_postfix(
                ordered_dict={
                    "Loss": loss.item()
                }
            )

    # Per-prompt loss
    for i, cate in enumerate(cates):
        print(f"{cate} loss: {net(xs[i], ys[i])['loss']}")

    # save_model("new", net)
    preds = net(xs, ys)["pred"]
    for i, cate in enumerate(cates):
        save_model_grid(f"last-{cate}", f"new-{cate}", preds[i])


def test_hyper_net():

    setup()

    cate_1, cate_2 = "hamburger", "pineapple"

    embed_1, grid_1 = load_embed(cate_1), load_grid(cate_1)
    embed_2, grid_2 = load_embed(cate_2), load_grid(cate_2)

    config = {
        "in_dim": embed_1.shape[0],
        "hid_dim": 3,
        "out_dim": grid_1.shape[0]
    }
    # model = load_model("last", config)

    print(f"Embedding Error {F.mse_loss(embed_1, embed_2)}")
    print(f"Densegrid Error {F.mse_loss(grid_1, grid_2)}, Max {(grid_1 - grid_2).abs().max()}")
    # print(f"Model Output Error {model(embed_1, grid_1)['loss']}")
    # print(f"Model Output Error {model(embed_2, grid_2)['loss']}")


def copy_hyper_net():

    setup()

    cates = ["hamburger", "pineapple"]
    embed, grid = load_embed(cates[0]), load_grid(cates[0])
    config = {
        "in_dim": embed.shape[0],
        "hid_dim": 3,
        "out_dim": grid.shape[0]
    }

    for cate in cates:
        model = load_model(cate, config)
        save_model(cate + "-new", model)


# test_hyper_net()
# train_hyper_net()
# copy_hyper_net()
