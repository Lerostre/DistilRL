import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import pytorch_lightning as pl
from typing import Dict, Optional, Iterable


class LightningBase(pl.LightningModule):
    """
    Base class for other pl models. Accepts any torch.optim optimizer
    and compatible schedulers. Does not support pl.Tuner usage!
    Args:
        optimizer: any torch optimizer
        optimizer_kwargs: optimizer parameters, such as lr
        scheduler: any scheduler compatible with torch optimizer
        scheduler_kwargs: scheduler parameters, such as warmup, num_training_steps etc
        save_hparams: flag for hyperparameters logging
    """

    def __init__(
        self,
        loss=nn.CrossEntropyLoss,
        optimizer: Optional[torch.optim.Optimizer] = optim.SGD,
        optimizer_kwargs: Optional[Dict[str, object]] = dict(lr=0.1),
        scheduler: Optional[object] = None,
        scheduler_kwargs: Optional[Dict[str, object]] = dict(),
        save_hparams: Optional[bool] = False,
    ):
        super().__init__()
        if save_hparams:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.loss = loss(reduction="none")

    def score_batch(self, batch):
        return Exception("Scoring should be implemented first")

    def step(
        self,
        batch,
        batch_idx,
        subset: Optional[str] = "train",
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=False,
    ):
        """
        Common step logic should be put here if necessary.
        Args:
            batch: batch to process, automatically fetched by pl.Trainer
            batch_idx: batch_idx to process, automatically fetched by pl.Trainer
            subset: step id for logic differentiation
        """
        res = self.score_batch(batch)
        if subset != "predict":
            for key, value in res.items():
                self.log(
                    f"{subset}_{key}",
                    value.detach().item(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=prog_bar,
                    logger=logger,
                )
        return res

    def training_step(
        self,
        batch,
        batch_idx,
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=False,
    ):
        """Training step. Log settings should be changed if needed."""
        return self.step(
            batch,
            batch_idx,
            "train",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=False,
    ):
        """Validation step. Log settings should be changed if needed."""
        return self.step(
            batch,
            batch_idx,
            "valid",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(
        self,
        batch,
        batch_idx,
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=False,
    ):
        """Test step. Log settings should be changed if needed."""
        return self.step(
            batch,
            batch_idx,
            "test",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
        on_step=False,
        on_epoch=False,
        prog_bar=False,
        logger=False,
    ):
        """Prediction step. Log settings should be changed if needed."""
        return self.step(batch, batch_idx, "predict")

    def configure_optimizers(self):
        """
        Optimizer and scheduler configuration.
        In most settings should not be tampered with
        """
        config = {}
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        config["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            config["lr_scheduler"] = scheduler

        return config


class BestValidCallback(pl.Callback):
    """
    Callback to save best validation result across epochs.
    Accepts any 'monitor' list of metric names, provided they were
    logged at validation_step. Stored at pl_module.best_valid
    """

    def __init__(self, monitor: Iterable[str] = ["valid_accuracy"]):
        self.monitor = monitor

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataloader_idx: int = 0,
    ):
        if not hasattr(pl_module, "best_valid"):
            pl_module.best_valid = dict()
        else:
            for metric in self.monitor:
                valid_metric_mean = trainer.callback_metrics[monitor]
                if valid_accuracy_mean > pl_module.best_valid[monitor]:
                    pl_module.best_valid[monitor] = valid_metric_mean


class ValidHistoryCallback(pl.Callback):

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if trainer.global_step != 0:
            valid_accuracy_step = trainer.callback_metrics[
                "valid_accuracy"
            ].item()
            try:
                pl_module.valid_history.append(valid_accuracy_step)
            except:
                pl_module.valid_history = [valid_accuracy_step]
