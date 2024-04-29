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
from torch_uncertainty.utils.distributions import dist_rearrange, squeeze_dist


class DepthRoutine(LightningModule):
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
        num_estimators: int = 1,
        optim_recipe: dict | Optimizer | None = None,
        format_batch_fn: nn.Module | None = None,
        num_image_plot: int = 4,
    ) -> None:
        super().__init__()
        _depth_routine_checks(num_estimators, output_dim)

        self.model = model
        self.output_dim = output_dim
        self.one_dim_depth = output_dim == 1
        self.probabilistic = probabilistic
        self.loss = loss
        self.num_estimators = num_estimators
        self.num_image_plot = num_image_plot

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
                "iMAE": MeanAbsoluteErrorInverse(),
                "iRMSE": MeanSquaredErrorInverse(squared=False),
                "d1": ThresholdAccuracy(power=1),
                "d2": ThresholdAccuracy(power=2),
                "d3": ThresholdAccuracy(power=3),
            },
            compute_groups=False,
        )

        self.val_metrics = depth_metrics.clone(prefix="val/")
        self.test_metrics = depth_metrics.clone(prefix="test/")

        if self.probabilistic:
            depth_prob_metrics = MetricCollection(
                {"NLL": DistributionNLL(reduction="mean")}
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
            if self.num_estimators == 1:
                pred = squeeze_dist(pred, -1)
        else:
            if self.num_estimators == 1:
                pred = pred.squeeze(-1)
        return pred

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, target = self.format_batch_fn(batch)
        if self.one_dim_depth:
            target = target.unsqueeze(1)

        dists = self.model(inputs)
        target = F.resize(
            target, dists.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        valid_mask = ~torch.isnan(target)
        loss = self.loss(dists[valid_mask], target[valid_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, target = batch
        if self.one_dim_depth:
            target = target.unsqueeze(1)
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = Independent(
                dist_rearrange(
                    preds, "(m b) c h w -> b m c h w", m=self.num_estimators
                ),
                1,
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            preds = mixture.mean
        else:
            preds = rearrange(
                preds, "(m b) c h w -> b m c h w", m=self.num_estimators
            )
            preds = preds.mean(dim=1)

        if batch_idx == 0:
            self._plot_depth(
                inputs[: self.num_image_plot, ...],
                preds[: self.num_image_plot, ...],
                target[: self.num_image_plot, ...],
                stage="val",
            )

        valid_mask = ~torch.isnan(target)
        self.val_metrics.update(preds[valid_mask], target[valid_mask])
        if self.probabilistic:
            self.val_prob_metrics.update(
                mixture[valid_mask], target[valid_mask]
            )

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

        inputs, target = batch
        if self.one_dim_depth:
            target = target.unsqueeze(1)
        preds = self.model(inputs)

        if self.probabilistic:
            ens_dist = dist_rearrange(
                preds, "(m b) c h w -> b m c h w", m=self.num_estimators
            )
            mix = Categorical(
                torch.ones(self.num_estimators, device=self.device)
            )
            mixture = MixtureSameFamily(mix, ens_dist)
            self.test_metrics.nll.update(mixture, target)
            preds = mixture.mean
        else:
            preds = rearrange(
                preds, "(m b) c h w -> b m c h w", m=self.num_estimators
            )
            preds = preds.mean(dim=1)

        if batch_idx == 0:
            num_images = (
                self.num_image_plot
                if self.num_image_plot < inputs.size(0)
                else inputs.size(0)
            )
            self._plot_depth(
                inputs[:num_images, ...],
                preds[:num_images, ...],
                target[:num_images, ...],
                stage="test",
            )

        valid_mask = ~torch.isnan(target)
        self.test_metrics.update(preds[valid_mask], target[valid_mask])
        if self.probabilistic:
            self.test_prob_metrics.update(
                mixture[valid_mask], target[valid_mask]
            )

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


def _depth_routine_checks(num_estimators: int, output_dim: int) -> None:
    if num_estimators < 1:
        raise ValueError(
            f"num_estimators must be positive, got {num_estimators}."
        )

    if output_dim < 1:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
