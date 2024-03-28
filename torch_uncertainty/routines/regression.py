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
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_uncertainty.metrics.regression.nll import DistributionNLL
from torch_uncertainty.utils.distributions import dist_rearrange, squeeze_dist


class RegressionRoutine(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        probabilistic: bool,
        loss: nn.Module,
        num_estimators: int = 1,
        optim_recipe: dict | Optimizer | None = None,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        r"""Routine for efficient training and testing on **regression tasks**
        using LightningModule.

        Args:
            model (torch.nn.Module): Model to train.
            output_dim (int): Number of outputs of the model.
            probabilistic (bool): Whether the model is probabilistic, i.e.,
                outputs a PyTorch distribution.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
            num_estimators (int, optional): The number of estimators for the
                ensemble. Defaults to ``1`` (single model).
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            format_batch_fn (torch.nn.Module, optional): The function to format the
                batch. Defaults to ``None``.

        Warning:
            If :attr:`probabilistic` is True, the model must output a `PyTorch
            distribution <https://pytorch.org/docs/stable/distributions.html>`_.

        Warning:
            You must define :attr:`optim_recipe` if you do not use
            the CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _regression_routine_checks(num_estimators, output_dim)

        self.model = model
        self.probabilistic = probabilistic
        self.output_dim = output_dim
        self.loss = loss
        self.num_estimators = num_estimators

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn

        reg_metrics = MetricCollection(
            {
                "MAE": MeanAbsoluteError(),
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
            },
            compute_groups=True,
        )

        self.val_metrics = reg_metrics.clone(prefix="reg_val/")
        self.test_metrics = reg_metrics.clone(prefix="reg_test/")

        if self.probabilistic:
            reg_prob_metrics = MetricCollection(
                {"NLL": DistributionNLL(reduction="mean")}
            )
            self.val_prob_metrics = reg_prob_metrics.clone(prefix="reg_val/")
            self.test_prob_metrics = reg_prob_metrics.clone(prefix="reg_test/")

        self.one_dim_regression = output_dim == 1

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
            if self.one_dim_regression:
                pred = squeeze_dist(pred, -1)
            if self.num_estimators == 1:
                pred = squeeze_dist(pred, -1)
        else:
            if self.one_dim_regression:
                pred = pred.squeeze(-1)
            if self.num_estimators == 1:
                pred = pred.squeeze(-1)
        return pred

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(batch)
        dists = self.model(inputs)

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)

        loss = self.loss(dists, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                dist_rearrange(
                    preds, "(m b) c -> b m c", m=self.num_estimators
                ),
                1,
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c -> b m c", m=self.num_estimators)
            preds = preds.mean(dim=1)

        self.val_metrics.update(preds, targets)
        if self.probabilistic:
            self.val_prob_metrics.update(mixture, targets)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
        if self.probabilistic:
            self.log_dict(
                self.val_prob_metrics.compute(),
            )
            self.val_prob_metrics.reset()

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
        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                dist_rearrange(
                    preds, "(m b) c -> b m c", m=self.num_estimators
                ),
                1,
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c -> b m c", m=self.num_estimators)
            preds = preds.mean(dim=1)

        self.test_metrics.update(preds, targets)
        if self.probabilistic:
            self.test_prob_metrics.update(mixture, targets)

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
        )
        self.test_metrics.reset()

        if self.probabilistic:
            self.log_dict(
                self.test_prob_metrics.compute(),
            )
            self.test_prob_metrics.reset()


def _regression_routine_checks(num_estimators: int, output_dim: int) -> None:
    if num_estimators < 1:
        raise ValueError(
            f"num_estimators must be positive, got {num_estimators}."
        )

    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
