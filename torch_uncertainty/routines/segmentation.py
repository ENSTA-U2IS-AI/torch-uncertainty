from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Accuracy, MetricCollection
from torchvision.transforms.v2 import functional as F

from torch_uncertainty.metrics import (
    CE,
    BrierScore,
    CategoricalNLL,
    MeanIntersectionOverUnion,
)


class SegmentationRoutine(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module,
        num_estimators: int = 1,
        optim_recipe: dict | Optimizer | None = None,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        """Routine for efficient training and testing on **segmentation tasks**
        using LightningModule.

        Args:
            model (torch.nn.Module): Model to train.
            num_classes (int): Number of classes in the segmentation task.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
            num_estimators (int, optional): The number of estimators for the
                ensemble. Defaults to ̀`1̀` (single model).
            optim_recipe (dict or Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            format_batch_fn (torch.nn.Module, optional): The function to format the
                batch. Defaults to ``None``.

        Warning:
            You must define :attr:`optim_recipe` if you do not use
            the CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _segmentation_routine_checks(num_estimators, num_classes)

        self.model = model
        self.num_classes = num_classes
        self.loss = loss
        self.num_estimators = num_estimators

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.optim_recipe = optim_recipe
        self.format_batch_fn = format_batch_fn

        # metrics
        seg_metrics = MetricCollection(
            {
                "Acc": Accuracy(task="multiclass", num_classes=num_classes),
                "ECE": CE(task="multiclass", num_classes=num_classes),
                "mIoU": MeanIntersectionOverUnion(num_classes=num_classes),
                "Brier": BrierScore(num_classes=num_classes),
                "NLL": CategoricalNLL(),
            },
            compute_groups=[["Acc", "mIoU"], ["ECE"], ["Brier"], ["NLL"]],
        )

        self.val_seg_metrics = seg_metrics.clone(prefix="seg_val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="seg_test/")

    def configure_optimizers(self) -> Optimizer | dict:
        return self.optim_recipe

    def forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def on_train_start(self) -> None:
        init_metrics = dict.fromkeys(self.val_seg_metrics, 0)
        init_metrics.update(dict.fromkeys(self.test_seg_metrics, 0))

        self.logger.log_hyperparams(self.hparams, init_metrics)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        img, target = batch
        img, target = self.format_batch_fn((img, target))
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(logits, "b c h w -> (b h w) c")
        target = target.flatten()
        valid_mask = target != 255
        loss = self.loss(logits[valid_mask], target[valid_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        img, target = batch
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(
            logits, "(m b) c h w -> (b h w) m c", m=self.num_estimators
        )
        probs_per_est = logits.softmax(dim=-1)
        probs = probs_per_est.mean(dim=1)
        target = target.flatten()
        valid_mask = target != 255
        self.val_seg_metrics.update(probs[valid_mask], target[valid_mask])

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        img, target = batch
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(
            logits, "(m b) c h w -> (b h w) m c", m=self.num_estimators
        )
        probs_per_est = logits.softmax(dim=-1)
        probs = probs_per_est.mean(dim=1)
        target = target.flatten()
        valid_mask = target != 255
        self.test_seg_metrics.update(probs[valid_mask], target[valid_mask])

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_seg_metrics.compute())
        self.val_seg_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_seg_metrics.compute())
        self.test_seg_metrics.reset()


def _segmentation_routine_checks(num_estimators: int, num_classes: int) -> None:
    if num_estimators < 1:
        raise ValueError(
            f"num_estimators must be positive, got {num_estimators}."
        )

    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
