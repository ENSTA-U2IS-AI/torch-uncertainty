import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
)
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_uncertainty.metrics.nll import DistributionNLL
from torch_uncertainty.utils.distributions import squeeze_dist, to_ensemble_dist


class RegressionRoutine(LightningModule):
    def __init__(
        self,
        probabilistic: bool,
        num_outputs: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int = 1,
        format_batch_fn: nn.Module | None = None,
        optimization_procedure=None,
    ) -> None:
        """Regression routine for PyTorch Lightning.

        Args:
            probabilistic (bool): Whether the model is probabilistic, i.e.,
                outputs a PyTorch distribution.
            num_outputs (int): The number of outputs of the model.
            model (nn.Module): The model to train.
            loss (type[nn.Module]): The loss function to use.
            num_estimators (int, optional): The number of estimators for the
                ensemble. Defaults to 1.
            format_batch_fn (nn.Module, optional): The function to format the
                batch. Defaults to None.
            optimization_procedure (optional): The optimization procedure
                to use. Defaults to None.

        Warning:
            If :attr:`probabilistic` is True, the model must output a `PyTorch
            distribution <https://pytorch.org/docs/stable/distributions.html>_`.
        """
        super().__init__()

        self.probabilistic = probabilistic
        self.model = model
        self.loss = loss

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.format_batch_fn = format_batch_fn

        reg_metrics = MetricCollection(
            {
                "mae": MeanAbsoluteError(),
                "mse": MeanSquaredError(squared=False),
            },
            compute_groups=False,
        )
        if self.probabilistic:
            reg_metrics["nll"] = DistributionNLL(reduction="mean")

        self.val_metrics = reg_metrics.clone(prefix="reg_val/")
        self.test_metrics = reg_metrics.clone(prefix="reg_test/")

        if num_estimators < 1:
            raise ValueError(
                f"num_estimators must be positive, got {num_estimators}."
            )
        self.num_estimators = num_estimators

        if num_outputs < 1:
            raise ValueError(
                f"num_outputs must be positive, got {num_outputs}."
            )
        self.num_outputs = num_outputs

        self.one_dim_regression = False
        if num_outputs == 1:
            self.one_dim_regression = True

        self.optimization_procedure = optimization_procedure

    def configure_optimizers(self):
        return self.optimization_procedure(self.model)

    def on_train_start(self) -> None:
        # hyperparameters for performances
        init_metrics = {k: 0 for k in self.val_metrics}
        init_metrics.update({k: 0 for k in self.test_metrics})

        if self.logger is not None:  # coverage: ignore
            self.logger.log_hyperparams(
                self.hparams,
                init_metrics,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        pred = self.model(inputs)
        if self.probabilistic:
            if self.one_dim_regression:
                pred = squeeze_dist(pred, -1)
            if self.num_estimators == 1:
                pred = squeeze_dist(pred, -1)
        return pred

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(batch)
        dists = self.model(inputs)

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)

        loss = self.criterion(dists, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        pred = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                to_ensemble_dist(pred, num_estimators=self.num_estimators), 1
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            pred = mixture.mean

        if self.one_dim_regression:
            print("one dim")
            targets = targets.unsqueeze(-1)

        self.val_metrics.mse.update(pred, targets)
        self.val_metrics.mae.update(pred, targets)
        if self.probabilistic:
            self.val_metrics.nll.update(mixture, targets)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

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
        pred = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                to_ensemble_dist(pred, num_estimators=self.num_estimators), 1
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            pred = mixture.mean

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)

        self.test_metrics.mse.update(pred, targets)
        self.test_metrics.mae.update(pred, targets)
        if self.probabilistic:
            self.test_metrics.nll.update(mixture, targets)

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
        )
        self.test_metrics.reset()
