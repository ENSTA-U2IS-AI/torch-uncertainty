from argparse import ArgumentParser, Namespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import (
    MetricCollection,
    Accuracy,
    CalibrationError,
    AUROC,
    AveragePrecision,
)

import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..metrics import Entropy, FPR95Metric, NegativeLogLikelihood


class ClassificationSingle(pl.LightningModule):
    def __init__(self, num_classes: int, *args, **kwargs) -> None:
        super().__init__()

        self.use_logits: bool = kwargs.get("use_logits", False)
        self.use_entropy: bool = kwargs.get("use_entropy", False)

        assert (
            self.use_logits + self.use_entropy + self.use_1v2
        ) <= 1, "You cannot choose more than one OOD criterion."

        # metrics
        cls_metrics = MetricCollection(
            {
                "nll": NegativeLogLikelihood(),
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "ece": CalibrationError(
                    task="multiclass", num_classes=num_classes
                ),
            },
            compute_groups=False,
        )

        ood_metrics = MetricCollection(
            {
                "fpr95": FPR95Metric(pos_label=1),
                "auroc": AUROC(task="binary"),
                "aupr": AveragePrecision(task="binary"),
            },
            compute_groups=[["auroc", "aupr"]],
        )

        self.val_metrics = cls_metrics.clone(prefix="hp/val_")
        self.test_cls_metrics = cls_metrics.clone(prefix="hp/test_")
        self.test_ood_metrics = ood_metrics.clone(prefix="hp/test_")

        self.test_entropy_id = Entropy()
        self.test_entropy_ood = Entropy()

    @property
    def criterion(self) -> nn.Module:
        raise NotImplementedError()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:
            self.logger.log_hyperparams(
                Namespace(**param),
                {
                    "hp/val_nll": 0,
                    "hp/val_acc": 0,
                    "hp/test_acc": 0,
                    "hp/test_acc_top5": 0,
                    "hp/test_nll": 0,
                    "hp/test_ece": 0,
                    "hp/test_entropy_id": 0,
                    "hp/test_entropy_ood": 0,
                    "hp/test_aupr": 0,
                    "hp/test_auroc": 0,
                    "hp/test_fpr95": 0,
                },
            )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        probs = F.sofmax(logits, dim=-1)
        self.val_metrics.update(probs, targets)
        self.log_dict(self.val_metrics.compute(), on_epoch=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        probs = F.sofmax(logits, dim=-1)
        confs, _ = probs.max(dim=-1)

        if self.use_logits:
            ood_values, _ = -logits.max(dim=-1)
        elif self.use_entropy:
            ood_values = torch.special.entr(probs).sum(dim=-1)
        else:
            ood_values = -confs

        if dataloader_idx == 0:
            self.test_cls_metrics.update(probs, targets)
            self.test_ood_metrics.update(ood_values, torch.zeros_like(targets))
            self.test_entropy_id(probs)
            self.log_dict(
                self.test_cls_metrics.compute(),
                on_epoch=True,
                add_dataloader_idx=False,
            )
            self.log_dict(
                self.test_ood_metrics.compute(),
                on_epoch=True,
                add_dataloader_idx=False,
            )
            self.log(
                "test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        else:
            self.test_ood_metrics(ood_values, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.log_dict(
                self.test_ood_metrics.compute(),
                on_epoch=True,
                add_dataloader_idx=False,
            )
            self.log(
                "test_entropy_ood",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        return parent_parser
