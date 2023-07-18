# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Any, List, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
        dist_estimation: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "optimization_procedure",
            ]
        )

        self.model = model
        self.loss = loss
        self.optimization_procedure = optimization_procedure
        self.dist_estimation = dist_estimation

        # metrics
        if dist_estimation:
            reg_metrics = MetricCollection(
                {
                    "mse": MeanSquaredError(squared=False),
                    "gnll": GaussianNegativeLogLikelihood(),
                },
                compute_groups=False,
            )
        else:
            reg_metrics = MetricCollection(
                {
                    "mse": MeanSquaredError(squared=False),
                },
                compute_groups=False,
            )

        self.val_metrics = reg_metrics.clone(prefix="hp/val_")
        self.test_metrics = reg_metrics.clone(prefix="hp/test_")

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

        if self.dist_estimation:
            means = logits[..., 0]
            vars = F.softplus(logits[..., 1])
            loss = self.criterion(means, targets, vars)
        else:
            loss = self.criterion(logits, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        if self.dist_estimation:
            means = logits[..., 0]
            vars = F.softplus(logits[..., 1])
            self.val_metrics.gnll.update(means, targets, vars)

            if means.ndim == 1:
                means = means.unsqueeze(-1)
        else:
            means = logits.squeeze(-1)

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
        if self.dist_estimation:
            means = logits[..., 0]
            vars = F.softplus(logits[..., 1])
            self.test_metrics.gnll.update(means, targets, vars)

            if means.ndim == 1:
                means = means.unsqueeze(-1)
        else:
            means = logits.squeeze(-1)

        self.test_metrics.mse.update(means, targets)

    def test_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(
            self.test_metrics.compute(),
        )
        self.test_metrics.reset()

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        return parent_parser


class RegressionEnsemble(RegressionSingle):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimization_procedure: Any,
        dist_estimation: bool,
        num_estimators: int,
        mode: Literal["mean", "mixture"],
        out_features: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            dist_estimation=dist_estimation,
            **kwargs,
        )

        if mode == "mixture":
            raise NotImplementedError(
                "Mixture of gaussians not implemented yet. Raise an issue if "
                "needed."
            )

        self.mode = mode
        self.num_estimators = num_estimators
        self.out_features = out_features

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch

        # eventual input repeat is done in the model
        targets = targets.repeat((self.num_estimators, 1))
        return super().training_step((inputs, targets), batch_idx)

    def validation_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.out_features == 1:
            logits = rearrange(
                logits, "(m b) dist -> b m dist", m=self.num_estimators
            )
        else:
            logits = rearrange(
                logits,
                "(m b) (f dist) -> b f m dist",
                m=self.num_estimators,
                f=self.out_features,
            )

        if self.mode == "mean":
            logits = logits.mean(dim=1)

        if self.dist_estimation:
            means = logits[..., 0]
            vars = F.softplus(logits[..., 1])
            self.val_metrics.gnll.update(means, targets, vars)
        else:
            means = logits

        self.val_metrics.mse.update(means, targets)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> None:
        if dataloader_idx != 0:
            raise NotImplementedError(
                "Regression OOD detection not implemented yet. Raise an issue "
                "if needed."
            )

        inputs, targets = batch
        logits = self.forward(inputs)

        if self.out_features == 1:
            logits = rearrange(
                logits, "(m b) dist -> b m dist", m=self.num_estimators
            )
        else:
            logits = rearrange(
                logits,
                "(m b) (f dist) -> b f m dist",
                m=self.num_estimators,
                f=self.out_features,
            )

        if self.mode == "mean":
            logits = logits.mean(dim=1)

        if self.dist_estimation:
            means = logits[..., 0]
            vars = F.softplus(logits[..., 1])
            self.test_metrics.gnll.update(means, targets, vars)
        else:
            means = logits

        self.test_metrics.mse.update(means, targets)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options:

        - ``--num_estimators``: sets :attr:`num_estimators`.
        """
        parent_parser = RegressionSingle.add_model_specific_args(parent_parser)
        parent_parser.add_argument(
            "--num_estimators",
            type=int,
            default=None,
            help="Number of estimators for ensemble",
        )
        return parent_parser
