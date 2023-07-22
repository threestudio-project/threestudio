import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from spec_norm import spectral_norm
from tqdm import tqdm

import pytorch_lightning as pl



class System(pl.LightningModule):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.lin1 = spectral_norm(nn.Linear(in_dim, hid_dim, bias=True))
        self.lin2 = spectral_norm(nn.Linear(hid_dim, out_dim, bias=False))
    
    def training_step(self, batch: dict) -> float:

        x = batch["x"]
        y = batch["y"]

        x = self.lin1(x)
        x = F.silu(x, inplace=True)
        x = self.lin2(x)
        
        loss = F.mse_loss(x, y)
        # print(f"Loss: {loss.item()}")
        return loss
    
    def configure_optimizers(self):
        ret = {
            "optimizer": Adam(self.parameters(), lr=1e-5, betas=[0.9, 0.999]),
        }
        return ret


class Data(Dataset):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_iter: int) -> None:
        super().__init__()
        self.dim = in_dim
        self.n_iter = n_iter
        self.mat1 = nn.Linear(in_dim, hid_dim, bias=True)
        self.mat2 = nn.Linear(hid_dim, out_dim, bias=False)
        for mod in [self.mat1, self.mat2]:
            for par in mod.parameters():
                par.requires_grad = False
    
    def __len__(self):
        return self.n_iter
    
    def __getitem__(self, index):
        x = torch.randn(1, self.dim)
        y = self.mat2(F.silu(self.mat1(x), inplace=True))
        batch = {
            "x": x,
            "y": y
        }
        return batch
    

class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = Data(**kwargs)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=1,  # type: ignore
            batch_size=batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.dataset, batch_size=1
        )
    

def main():
    config = {
        "n_iter": 100,
        "in_dim": 77 * 4096,
        "hid_dim": 32,
        # "out_dim": 1034048,
        "out_dim": 1634048,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    system = System(**config)
    datamodule = DataModule(**config)

    trainer = pl.Trainer(max_steps=config["n_iter"])
    trainer.fit(system, datamodule=datamodule)

    avg_loss = 0
    for batch in tqdm(datamodule.train_dataloader()):

        loss = system.training_step(batch)
        avg_loss += loss.item()
    
    avg_loss /= config["n_iter"]
    print(f"AVG Loss {avg_loss}")


main()

