import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
)
from torch.optim import Optimizer
from torchmetrics import MeanSquaredError, MetricCollection

from torch_uncertainty.metrics import (
    DistributionNLL,
    Log10,
    MeanGTRelativeAbsoluteError,
    MeanGTRelativeSquaredError,
    MeanSquaredLogError,
    SILog,
    ThresholdAccuracy,
)
from torch_uncertainty.utils.distributions import dist_rearrange, squeeze_dist


class DepthRoutine(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        probabilistic: bool,
        loss: nn.Module,
        num_estimators: int = 1,
        optim_recipe: dict | Optimizer | None = None,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        _depth_routine_checks(num_estimators)

        self.model = model
        self.probabilistic = probabilistic
        self.loss = loss
        self.num_estimators = num_estimators

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn

        depth_metrics = MetricCollection(
            {
                "SILog": SILog(),
                "log10": Log10(),
                "ARE": MeanGTRelativeAbsoluteError(),
                "RSRE": MeanGTRelativeSquaredError(squared=False),
                "RMSE": MeanSquaredError(squared=False),
                "RMSELog": MeanSquaredLogError(squared=False),
                "d1": ThresholdAccuracy(power=1),
                "d2": ThresholdAccuracy(power=2),
                "d3": ThresholdAccuracy(power=3),
            },
            compute_groups=True,
        )

        self.val_metrics = depth_metrics.clone(prefix="depth_val/")
        self.test_metrics = depth_metrics.clone(prefix="depth_test/")

        if self.probabilistic:
            depth_prob_metrics = MetricCollection(
                {"NLL": DistributionNLL(reduction="mean")}
            )
            self.val_prob_metrics = depth_prob_metrics.clone(
                prefix="depth_val/"
            )
            self.test_prob_metrics = depth_prob_metrics.clone(
                prefix="depth_test/"
            )

    def configure_optimizers(self) -> Optimizer | dict:
        return self.optim_recipe

    def on_train_start(self) -> None:
        init_metrics = dict.fromkeys(self.val_metrics, 0)
        init_metrics.update(dict.fromkeys(self.test_metrics, 0))

        if self.probabilistic:
            init_metrics.update(dict.fromkeys(self.val_prob_metrics, 0))
            init_metrics.update(dict.fromkeys(self.test_prob_metrics, 0))

        if self.logger is not None:  # coverage: ignore
            self.logger.log_hyperparams(
                self.hparams,
                init_metrics,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of the routine.

        The forward pass automatically squeezes the output if the regression
        is one-dimensional and if the routine contains a single model.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        pred = self.model(inputs)
        if self.probabilistic:
            if self.num_estimators == 1:
                pred = squeeze_dist(pred, -1)
        else:
            if self.num_estimators == 1:
                pred = pred.squeeze(-1)
        return pred

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(batch)
        dists = self.model(inputs)
        targets = targets.unsqueeze(-1)
        loss = self.loss(dists, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                dist_rearrange(preds, "(m b) -> b m", m=self.num_estimators), 1
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) -> b m", m=self.num_estimators)
            preds = preds.mean(dim=1)

        self.val_metrics.update(preds, targets)
        if self.probabilistic:
            self.val_prob_metrics.update(mixture, targets)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0:
            raise NotImplementedError(
                "Depth OOD detection not implemented yet. Raise an issue "
                "if needed."
            )

        inputs, targets = batch
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = dist_rearrange(
                preds, "(m b) -> b m", m=self.num_estimators
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            self.test_metrics.nll.update(mixture, targets)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b)-> b m", m=self.num_estimators)
            preds = preds.mean(dim=1)

        self.test_metrics.update(preds, targets)
        if self.probabilistic:
            self.test_prob_metrics.update(mixture, targets)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
        if self.probabilistic:
            self.log_dict(
                self.val_prob_metrics.compute(),
                sync_dist=True,
            )
            self.val_prob_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
            sync_dist=True,
        )
        self.test_metrics.reset()
        if self.probabilistic:
            self.log_dict(
                self.test_prob_metrics.compute(),
                sync_dist=True,
            )
            self.test_prob_metrics.reset()


def _depth_routine_checks(num_estimators: int) -> None:
    if num_estimators < 1:
        raise ValueError(
            f"num_estimators must be positive, got {num_estimators}."
        )
