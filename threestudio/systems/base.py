import re
from dataclasses import dataclass, field
import pytorch_lightning as pl
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.config import parse_structured
from threestudio.utils.base import Updateable
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *
from threestudio.utils.misc import load_module_weights, C

class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loss: dict = field(default_factory=dict)
        freq: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self.configure()
        if self.cfg.weights is not None:
            self.load_state_dict(load_module_weights(self.cfg.weights, ignore_modules=self.cfg.weights_ignore_modules, map_location='cpu'), strict=False)
    
    def configure(self, *args, **kwargs) -> None:
        pass
    
    def C(self, value: Any) -> float:
        return C(value, self.current_epoch, self.global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            'optimizer': optim,
        }
        if self.cfg.scheduler is not None:
            ret.update({
                'lr_scheduler': parse_scheduler(self.cfg.scheduler, optim),
            })
        return ret
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
        
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def on_validation_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):        
        raise NotImplementedError
    
    def on_test_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """    

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, 'train')
        self.do_update_step(self.current_epoch, self.global_step)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, 'validation')
        self.do_update_step(self.current_epoch, self.global_step)
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, 'test')
        self.do_update_step(self.current_epoch, self.global_step)

    def update_step(self, epoch: int, global_step: int):
        pass
