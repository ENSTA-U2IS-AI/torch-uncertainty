from argparse import ArgumentParser
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
from torchmetrics import MeanSquaredError, MetricCollection

from torch_uncertainty.metrics.nll import GaussianNegativeLogLikelihood


class RegressionSingle(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: type[nn.Module],
        optimization_procedure: Any,
        dist_estimation: int,
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

        # metrics
        if isinstance(dist_estimation, int):
            if dist_estimation <= 0:
                raise ValueError(
                    "Expected the argument ``dist_estimation`` to be integer "
                    f" larger than 0, but got {dist_estimation}."
                )
        else:
            raise TypeError(
                "Expected the argument ``dist_estimation`` to be integer, but "
                f"got {type(dist_estimation)}"
            )

        out_features = list(self.model.parameters())[-1].size(0)
        if dist_estimation > out_features:
            raise ValueError(
                "Expected argument ``dist_estimation`` to be an int lower or "
                f"equal than the size of the output layer, but got "
                f"{dist_estimation} and {out_features}."
            )

        self.dist_estimation = dist_estimation

        if dist_estimation in (4, 2):
            reg_metrics = MetricCollection(
                {
                    "mse": MeanSquaredError(squared=True),
                    "gnll": GaussianNegativeLogLikelihood(),
                },
                compute_groups=False,
            )
        else:
            reg_metrics = MetricCollection(
                {
                    "mse": MeanSquaredError(squared=True),
                },
                compute_groups=False,
            )

        self.val_metrics = reg_metrics.clone(prefix="reg_val/")
        self.test_metrics = reg_metrics.clone(prefix="reg_test/")

    def configure_optimizers(self) -> Any:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(inputs)

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.dist_estimation == 4:
            means, v, alpha, beta = logits.split(1, dim=-1)
            v = F.softplus(v)
            alpha = 1 + F.softplus(alpha)
            beta = F.softplus(beta)
            loss = self.criterion(means, v, alpha, beta, targets)
        elif self.dist_estimation == 2:
            means = logits[..., 0]
            variances = F.softplus(logits[..., 1])
            loss = self.criterion(means, targets, variances)
        else:
            loss = self.criterion(logits, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        if self.dist_estimation == 4:
            means = logits[..., 0]
            alpha = 1 + F.softplus(logits[..., 2])
            beta = F.softplus(logits[..., 3])
            variances = beta / (alpha - 1)
            self.val_metrics.gnll.update(means, targets, variances)

            targets = targets.view(means.size())
        elif self.dist_estimation == 2:
            means = logits[..., 0]
            variances = F.softplus(logits[..., 1])
            self.val_metrics.gnll.update(means, targets, variances)

            if means.ndim == 1:
                means = means.unsqueeze(-1)
        else:
            means = logits.squeeze(-1)

        self.val_metrics.mse.update(means, targets)

    def validation_epoch_end(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]
    ) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.dist_estimation == 4:
            means = logits[..., 0]
            alpha = 1 + F.softplus(logits[..., 2])
            beta = F.softplus(logits[..., 3])
            variances = beta / (alpha - 1)
            self.test_metrics.gnll.update(means, targets, variances)

            targets = targets.view(means.size())
        elif self.dist_estimation == 2:
            means = logits[..., 0]
            variances = F.softplus(logits[..., 1])
            self.test_metrics.gnll.update(means, targets, variances)

            if means.ndim == 1:
                means = means.unsqueeze(-1)
        else:
            means = logits.squeeze(-1)

        self.test_metrics.mse.update(means, targets)

    def test_epoch_end(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]
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
        loss: type[nn.Module],
        optimization_procedure: Any,
        dist_estimation: int,
        num_estimators: int,
        mode: Literal["mean", "mixture"],
        out_features: int | None = 1,
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
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch

        # eventual input repeat is done in the model
        targets = targets.repeat((self.num_estimators, 1))
        return super().training_step((inputs, targets), batch_idx)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
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

        if self.dist_estimation == 2:
            means = logits[..., 0]
            variances = F.softplus(logits[..., 1])
            self.val_metrics.gnll.update(means, targets, variances)
        else:
            means = logits

        self.val_metrics.mse.update(means, targets)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = 0,
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

        if self.dist_estimation == 2:
            means = logits[..., 0]
            variances = F.softplus(logits[..., 1])
            self.test_metrics.gnll.update(means, targets, variances)
        else:
            means = logits

        self.test_metrics.mse.update(means, targets)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options.

        Adds:
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
