from typing import Literal

import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanSquaredError, MetricCollection

from torch_uncertainty.metrics.nll import GaussianNegativeLogLikelihood


class RegressionRoutine(LightningModule):
    def __init__(
        self,
        dist_estimation: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int,
        mode: Literal["mean", "mixture"],
        out_features: int | None = 1,
    ) -> None:
        print("Regression is Work in progress. Raise an issue if interested.")
        super().__init__()

        self.model = model
        self.loss = loss
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

        if mode == "mixture":
            raise NotImplementedError(
                "Mixture of gaussians not implemented yet. Raise an issue if "
                "needed."
            )

        self.mode = mode
        self.num_estimators = num_estimators
        self.out_features = out_features

    def on_train_start(self) -> None:
        # hyperparameters for performances
        init_metrics = {k: 0 for k in self.val_metrics}
        init_metrics.update({k: 0 for k in self.test_metrics})

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model.forward(inputs)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch

        # eventual input repeat is done in the model
        targets = targets.repeat((self.num_estimators, 1))

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
        self, batch: tuple[Tensor, Tensor], batch_idx: int
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
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
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

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(
            self.test_metrics.compute(),
        )
        self.test_metrics.reset()
