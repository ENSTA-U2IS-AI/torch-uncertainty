import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Distribution,
    MixtureSameFamily,
)
from torch.optim import Optimizer
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.metrics import (
    DistributionNLL,
)
from torch_uncertainty.models import (
    EPOCH_UPDATE_MODEL,
    STEP_UPDATE_MODEL,
)
from torch_uncertainty.utils.distributions import (
    dist_rearrange,
    dist_size,
    dist_squeeze,
)


class RegressionRoutine(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        probabilistic: bool,
        loss: nn.Module,
        is_ensemble: bool = False,
        optim_recipe: dict | Optimizer | None = None,
        eval_shift: bool = False,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        r"""Routine for training & testing on **regression** tasks.

        Args:
            model (torch.nn.Module): Model to train.
            output_dim (int): Number of outputs of the model.
            probabilistic (bool): Whether the model is probabilistic, i.e.,
                outputs a PyTorch distribution.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
            is_ensemble (bool, optional): Whether the model is an ensemble.
                Defaults to ``False``.
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
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
        _regression_routine_checks(output_dim)
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue "
                "if needed."
            )

        self.model = model
        self.probabilistic = probabilistic
        self.output_dim = output_dim
        self.loss = loss
        self.is_ensemble = is_ensemble
        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn

        reg_metrics = MetricCollection(
            {
                "reg/MAE": MeanAbsoluteError(),
                "reg/MSE": MeanSquaredError(squared=True),
                "reg/RMSE": MeanSquaredError(squared=False),
            },
            compute_groups=True,
        )

        self.val_metrics = reg_metrics.clone(prefix="val/")
        self.test_metrics = reg_metrics.clone(prefix="test/")

        if self.probabilistic:
            reg_prob_metrics = MetricCollection(
                {"reg/NLL": DistributionNLL(reduction="mean")}
            )
            self.val_prob_metrics = reg_prob_metrics.clone(prefix="val/")
            self.test_prob_metrics = reg_prob_metrics.clone(prefix="test/")

        self.one_dim_regression = output_dim == 1

    def configure_optimizers(self) -> Optimizer | dict:
        return self.optim_recipe

    def on_train_start(self) -> None:
        if self.logger is not None:  # coverage: ignore
            self.logger.log_hyperparams(
                self.hparams,
            )

    def on_validation_start(self) -> None:
        if self.needs_epoch_update and not self.trainer.sanity_checking:
            self.model.update_wrapper(self.current_epoch)
            if hasattr(self.model, "need_bn_update"):
                self.model.bn_update(
                    self.trainer.train_dataloader, device=self.device
                )

    def on_test_start(self) -> None:
        if hasattr(self.model, "need_bn_update"):
            self.model.bn_update(
                self.trainer.train_dataloader, device=self.device
            )

    def forward(self, inputs: Tensor) -> Tensor | Distribution:
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
                pred = dist_squeeze(pred, -1)
            if not self.is_ensemble:
                pred = dist_squeeze(pred, -1)
        else:
            if self.one_dim_regression:
                pred = pred.squeeze(-1)
            if not self.is_ensemble:
                pred = pred.squeeze(-1)
        return pred

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(batch)

        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)

        if isinstance(self.loss, ELBOLoss):
            loss = self.loss(inputs, targets)
        else:
            dists = self.model(inputs)
            loss = self.loss(dists, targets)

        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        batch_size = targets.size(0)
        targets = rearrange(targets, "b c -> (b c)")
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = dist_rearrange(preds, "(m b) c -> (b c) m", b=batch_size)
            mix = Categorical(
                torch.ones(
                    dist_size(preds)[0] // batch_size, device=self.device
                )
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c -> (b c) m", b=batch_size)
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
                "Regression OOD detection not implemented yet. Raise an issue "
                "if needed."
            )

        inputs, targets = batch
        if self.one_dim_regression:
            targets = targets.unsqueeze(-1)
        batch_size = targets.size(0)
        targets = rearrange(targets, "b c -> (b c)")
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = dist_rearrange(preds, "(m b) c -> (b c) m", b=batch_size)
            mix = Categorical(
                torch.ones(
                    dist_size(preds)[0] // batch_size, device=self.device
                )
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c -> (b c) m", b=batch_size)
            preds = preds.mean(dim=1)

        self.test_metrics.update(preds, targets)
        if self.probabilistic:
            self.test_prob_metrics.update(mixture, targets)

    def on_validation_epoch_end(self) -> None:
        res_dict = self.val_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        self.log(
            "RMSE",
            res_dict["val/reg/RMSE"],
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.val_metrics.reset()
        if self.probabilistic:
            self.log_dict(self.val_prob_metrics.compute(), sync_dist=True)
            self.val_prob_metrics.reset()

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


def _regression_routine_checks(output_dim: int) -> None:
    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
