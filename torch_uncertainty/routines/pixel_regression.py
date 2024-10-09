from typing import Literal

import matplotlib.cm as cm
import torch
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
)
from torch.optim import Optimizer
from torchmetrics import MeanSquaredError, MetricCollection
from torchvision.transforms.v2 import functional as F
from torchvision.utils import make_grid

from torch_uncertainty.metrics import (
    DistributionNLL,
    Log10,
    MeanAbsoluteErrorInverse,
    MeanGTRelativeAbsoluteError,
    MeanGTRelativeSquaredError,
    MeanSquaredErrorInverse,
    MeanSquaredLogError,
    SILog,
    ThresholdAccuracy,
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


class PixelRegressionRoutine(LightningModule):
    inv_norm_params = {
        "mean": [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        "std": [1 / 0.229, 1 / 0.224, 1 / 0.255],
    }

    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        probabilistic: bool,
        loss: nn.Module,
        is_ensemble: bool = False,
        format_batch_fn: nn.Module | None = None,
        optim_recipe: dict | Optimizer | None = None,
        eval_shift: bool = False,
        num_image_plot: int = 4,
        log_plots: bool = False,
    ) -> None:
        """Routine for training & testing on **pixel regression** tasks.

        Args:
            model (nn.Module): Model to train.
            output_dim (int): Number of outputs of the model.
            probabilistic (bool): Whether the model is probabilistic, i.e.,
                outputs a PyTorch distribution.
            loss (nn.Module): Loss function to optimize the :attr:`model`.
            is_ensemble (bool, optional): Whether the model is an ensemble.
                Defaults to ``False``.
            optim_recipe (dict or Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            format_batch_fn (nn.Module, optional): The function to format the
                batch. Defaults to ``None``.
            num_image_plot (int, optional): Number of images to plot. Defaults to ``4``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
        """
        super().__init__()
        _depth_routine_checks(output_dim, num_image_plot, log_plots)
        if eval_shift:
            raise NotImplementedError(
                "Distribution shift evaluation not implemented yet. Raise an issue "
                "if needed."
            )

        self.model = model
        self.output_dim = output_dim
        self.one_dim_depth = output_dim == 1
        self.probabilistic = probabilistic
        self.loss = loss
        self.num_image_plot = num_image_plot
        self.is_ensemble = is_ensemble
        self.log_plots = log_plots

        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn

        depth_metrics = MetricCollection(
            {
                "reg/SILog": SILog(),
                "reg/log10": Log10(),
                "reg/ARE": MeanGTRelativeAbsoluteError(),
                "reg/RSRE": MeanGTRelativeSquaredError(squared=False),
                "reg/RMSE": MeanSquaredError(squared=False),
                "reg/RMSELog": MeanSquaredLogError(squared=False),
                "reg/iMAE": MeanAbsoluteErrorInverse(),
                "reg/iRMSE": MeanSquaredErrorInverse(squared=False),
                "reg/d1": ThresholdAccuracy(power=1),
                "reg/d2": ThresholdAccuracy(power=2),
                "reg/d3": ThresholdAccuracy(power=3),
            },
            compute_groups=False,
        )

        self.val_metrics = depth_metrics.clone(prefix="val/")
        self.test_metrics = depth_metrics.clone(prefix="test/")

        if self.probabilistic:
            depth_prob_metrics = MetricCollection(
                {"reg/NLL": DistributionNLL(reduction="mean")}
            )
            self.val_prob_metrics = depth_prob_metrics.clone(prefix="val/")
            self.test_prob_metrics = depth_prob_metrics.clone(prefix="test/")

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
            if not self.is_ensemble:
                pred = dist_squeeze(pred, -1)
        else:
            if not self.is_ensemble:
                pred = pred.squeeze(-1)
        return pred

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, target = self.format_batch_fn(batch)
        if self.one_dim_depth:
            target = target.unsqueeze(1)

        dists = self.model(inputs)
        if self.probabilistic:
            out_shape = dist_size(dists)[-2:]
        else:
            out_shape = dists.shape[-2:]
        target = F.resize(
            target, out_shape, interpolation=F.InterpolationMode.NEAREST
        )
        padding_mask = torch.isnan(target)
        if self.probabilistic:
            loss = self.loss(dists, target, padding_mask)
        else:
            loss = self.loss(dists[padding_mask], target[padding_mask])

        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        if self.one_dim_depth:
            targets = targets.unsqueeze(1)
        batch_size = targets.size(0)
        targets = rearrange(targets, "b c h w -> (b c h w)")
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                dist_rearrange(
                    preds, "(m b) c h w -> (b c h w) m", b=batch_size
                ),
                0,
            )
            mix = Categorical(
                torch.ones(
                    (dist_size(preds)[0] // batch_size), device=self.device
                )
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c h w -> (b c h w) m", b=batch_size)
            preds = preds.mean(dim=1)

        if batch_idx == 0 and self.log_plots:
            self._plot_depth(
                inputs[: self.num_image_plot, ...],
                preds[: self.num_image_plot, ...],
                targets[: self.num_image_plot, ...],
                stage="val",
            )

        padding_mask = torch.isnan(targets)
        self.val_metrics.update(preds[padding_mask], targets[padding_mask])
        if self.probabilistic:
            self.val_prob_metrics.update(mixture, targets, padding_mask)

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
        if self.one_dim_depth:
            targets = targets.unsqueeze(1)
        batch_size = targets.size(0)
        targets = rearrange(targets, "b c h w -> (b c h w)")
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = dist_rearrange(
                preds, "(m b) c h w -> (b c h w) m", b=batch_size
            )
            mix = Categorical(
                torch.ones(
                    (dist_size(preds)[0] // batch_size), device=self.device
                )
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(preds, "(m b) c h w -> (b c h w) m", b=batch_size)
            preds = preds.mean(dim=1)

        if batch_idx == 0 and self.log_plots:
            num_images = (
                self.num_image_plot
                if self.num_image_plot < inputs.size(0)
                else inputs.size(0)
            )
            self._plot_depth(
                inputs[:num_images, ...],
                preds[:num_images, ...],
                targets[:num_images, ...],
                stage="test",
            )

        padding_mask = torch.isnan(targets)
        self.test_metrics.update(preds[padding_mask], targets[padding_mask])
        if self.probabilistic:
            self.test_prob_metrics.update(mixture, targets, padding_mask)

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

    def _plot_depth(
        self,
        inputs: Tensor,
        preds: Tensor,
        target: Tensor,
        stage: Literal["val", "test"],
    ) -> None:
        if (
            self.logger is not None
            and isinstance(self.logger, TensorBoardLogger)
            and self.one_dim_depth
        ):
            all_imgs = []
            for i in range(inputs.size(0)):
                img = F.normalize(inputs[i, ...].cpu(), **self.inv_norm_params)
                pred = colorize(
                    preds[i, 0, ...].cpu(), vmin=0, vmax=self.model.max_depth
                )
                tgt = colorize(
                    target[i, 0, ...].cpu(), vmin=0, vmax=self.model.max_depth
                )
                all_imgs.extend([img, pred, tgt])

            self.logger.experiment.add_image(
                f"{stage}/samples",
                make_grid(torch.stack(all_imgs, dim=0), nrow=3),
                self.current_epoch,
            )


def colorize(
    value: Tensor,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
):
    """Colorize a tensor of depth values.

    Args:
        value (Tensor): The tensor of depth values.
        vmin (float, optional): The minimum depth value. Defaults to None.
        vmax (float, optional): The maximum depth value. Defaults to None.
        cmap (str, optional): The colormap to use. Defaults to 'magma'.
    """
    vmin = value.min().item() if vmin is None else vmin
    vmax = value.max().item() if vmax is None else vmax
    if vmin == vmax:
        return torch.zeros_like(value)
    value = (value - vmin) / (vmax - vmin)
    cmapper = cm.get_cmap(cmap)
    value = cmapper(value.numpy(), bytes=True)
    img = value[:, :, :3]
    return torch.as_tensor(img).permute(2, 0, 1).float() / 255.0


def _depth_routine_checks(
    output_dim: int, num_image_plot: int, log_plots: bool
) -> None:
    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
    if num_image_plot < 1 and log_plots:
        raise ValueError(
            f"num_image_plot must be positive, got {num_image_plot}."
        )
