import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


x = torch.randn(1, 77 * 4096).float()

linear1 = spectral_norm(nn.Linear(77 * 4096, 32, bias=True))
linear2 = spectral_norm(nn.Linear(32, 12599920, bias=False))

print(x.shape)
x = linear1(x)
print(x.shape)
x = F.silu(x, inplace=True)
print(x.shape)
x = linear2(x)
print(x.shape)

