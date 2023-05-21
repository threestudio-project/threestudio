import torch
import torch_optimizer as toptim
from torch import nn
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lienar = nn.Linear(3, 2)
        self.proj = nn.Linear(3, 3)

    def forward(self, x):
        inp = self.proj(x)
        out, grad = self.compute_norm_density(inp)
        return out, grad, inp

    def compute_norm_density(self, inp):
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        inp.requires_grad_(True)
        out = self.lienar(inp)
        grad = torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]
        torch.set_grad_enabled(grad_enabled)
        return out, grad


model = Model().cuda()

optim = toptim.Adahessian(model.parameters(), lr=1.0e-2, hessian_power=1.0)

for i in tqdm(range(10000)):
    print(torch.cuda.memory_allocated())
    x = torch.rand(100, 3).cuda()
    optim.zero_grad(set_to_none=True)
    out, grad, inp = model(x)
    loss = (grad.norm(dim=-1) - 1).abs().mean() + (out.norm(dim=-1) - 1).abs().mean()
    loss.backward(create_graph=True)
    optim.step()

# optim = toptim.Adahessian(model.parameters(), lr=1.0e-2, hessian_power=1.0)

# for i in tqdm(range(10000)):
#     print(torch.cuda.memory_allocated())
#     xy = torch.rand(100, 3).cuda()
#     xy.requires_grad = True
#     optim.zero_grad(set_to_none=True)
#     out = model(xy)
#     # grad = torch.autograd.grad(out, xy, torch.ones_like(out), create_graph=True)[0]
#     # loss push grad should to have a norm 1
#     loss = (out.norm(dim=-1) - 1).abs().mean()
#     loss.backward(create_graph=True)
#     optim.step()

# optim = torch.optim.Adam(model.parameters(), lr=1.0e-2)

# for i in tqdm(range(10000)):
#     print(torch.cuda.memory_allocated())
#     xy = torch.rand(100, 3).cuda()
#     xy.requires_grad = True
#     optim.zero_grad(set_to_none=True)
#     out = model(xy)
#     grad = torch.autograd.grad(out, xy, torch.ones_like(out), create_graph=True)[0]
#     # loss push grad should to have a norm 1
#     loss = (grad.norm(dim=-1) - 1).abs().mean() + (out.norm(dim=-1) - 1).abs().mean()
#     loss.backward(create_graph=True)
#     optim.step()
