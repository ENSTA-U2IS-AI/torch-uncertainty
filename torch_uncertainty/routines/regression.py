# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torchmetrics import MeanSquaredError, MetricCollection

from ..metrics.nll import GaussianNegativeLogLikelihood


# fmt:on
class RegressionSingle(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimization_procedure: Any,
        **kwargs,
    ) -> None:
        super().__init__()

        # model
        self.model = model
        # loss
        self.loss = loss
        # optimization procedure
        self.optimization_procedure = optimization_procedure
        # metrics
        reg_metrics = MetricCollection(
            {
                "mse": MeanSquaredError(squared=False),
                "gnll": GaussianNegativeLogLikelihood(),
            },
            compute_groups=False,
        )

        self.val_metrics = reg_metrics.clone(prefix="hp/val_")
        self.test_reg_metrics = reg_metrics.clone(prefix="hp/test_")

    def configure_optimizers(self) -> Any:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model.forward(input)

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:
            self.logger.log_hyperparams(
                Namespace(**param),
                {
                    "hp/val_mse": 0,
                    "hp/val_gnll": 0,
                },
            )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        logits = self.forward(inputs)
        means = logits[:, 0]
        vars = F.softplus(logits[:, 1])
        loss = self.criterion(means, targets, vars)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        means = logits[:, 0]
        vars = F.softplus(logits[:, 1])

        self.val_metrics.gnll.update(means, targets, vars)
        self.val_metrics.mse.update(means, targets)

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        means = logits[:, 0]
        vars = F.softplus(logits[:, 1])

        self.test_reg_metrics.gnll.update(means, targets, vars)
        self.test_reg_metrics.mse.update(means, targets)

    def test_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(
            self.test_reg_metrics.compute(),
        )
        self.test_reg_metrics.reset()

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        return parent_parser