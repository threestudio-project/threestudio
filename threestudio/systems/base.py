from dataclasses import dataclass, field

import pytorch_lightning as pl

from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import C, cleanup, load_module_weights
from threestudio.utils.saving import SaverMixin
from threestudio.utils.typing import *


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loss: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False

    cfg: Config

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resume_eval: bool = False
        self._resume_eval_status: dict = {"global_step": 0, "current_epoch": 0}
        self.configure()
        if self.cfg.weights is not None:
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.weights,
                ignore_modules=self.cfg.weights_ignore_modules,
                map_location="cpu",
            )
            self.load_state_dict(state_dict, strict=False)
            self.do_update_step(
                epoch, global_step, on_load_weights=True
            )  # restore states
        self.post_configure()

    def set_resume_status(self, current_epoch: int, global_step: int):
        self._resume_eval = True
        self._resume_eval_status["current_epoch"] = current_epoch
        self._resume_eval_status["global_step"] = global_step

    @property
    def true_global_step(self):
        if self._resume_eval:
            return self._resume_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resume_eval:
            return self._resume_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_validation_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_test_epoch_end(self):
        """
        Gather metrics from all devices, compute mean.
        Purge repeated results using data index.
        """
        pass

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.preprocess_data(batch, "train")
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        """
        pass
