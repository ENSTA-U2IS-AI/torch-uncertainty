from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import MeanSquaredError, MetricCollection

from torch_uncertainty.metrics.nll import DistributionNLL


class RegressionRoutine(LightningModule):
    def __init__(
        self,
        num_features: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.format_batch_fn = format_batch_fn

        reg_metrics = MetricCollection(
            {
                "mse": MeanSquaredError(squared=True),
                "nll": DistributionNLL(),
            },
            compute_groups=False,
        )
        self.val_metrics = reg_metrics.clone(prefix="reg_val/")
        self.test_metrics = reg_metrics.clone(prefix="reg_test/")

        self.num_estimators = num_estimators

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

        loss = self.criterion(logits, targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch

        dists = self.forward(inputs)

        self.val_metrics.mse.update(dists.loc, targets)
        self.val_metrics.nll.update(dists, targets)

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
        dists = self.forward(inputs)

        self.test_metrics.mse.update(dists.loc, targets)
        self.test_metrics.nll.update(dists, targets)

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
        )
        self.test_metrics.reset()
