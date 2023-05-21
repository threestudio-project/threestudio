import tinycudann as tcnn
import torch
import torch_optimizer as toptim
from tqdm import tqdm

from threestudio.models.networks import ProgressiveBandHashGrid
from threestudio.utils.misc import get_rank

in_channels = 3
pos_encoding_config = {
    "otype": "ProgressiveBandHashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 1.447269237440378,
    "start_level": 8,
    "start_step": 2000,
    "update_steps": 500,
}
encoding = ProgressiveBandHashGrid(
    in_channels, pos_encoding_config, dtype=torch.float32
)

with torch.no_grad():
    xy_eval = torch.randn(2, in_channels).cuda()
    print(encoding(xy_eval))

# optim = torch.optim.Adam(encoding.parameters(), lr=1e-3)
# optim = toptim.Adahessian(encoding.parameters(), lr=1e-2, hessian_power=1.0)
optim = torch.optim.LBFGS(encoding.parameters(), lr=1)

for i in tqdm(range(100)):
    encoding.update_step(0, i)

    def closure():
        xy = torch.randn(16, in_channels).cuda()
        optim.zero_grad()
        xy = encoding(xy)
        loss = (xy - 1.0).abs().mean()
        loss.backward()
        return loss

    optim.step(closure)

with torch.no_grad():
    xy_eval = torch.randn(2, in_channels).cuda()
    print(encoding(xy_eval))
