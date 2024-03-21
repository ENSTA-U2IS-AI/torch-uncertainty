from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import Accuracy, MetricCollection

from torch_uncertainty.metrics import MeanIntersectionOverUnion


class SegmentationRoutine(LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int = 1,
        optim_recipe=None,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.num_classes = num_classes
        self.model = model
        self.loss = loss
        self.num_estimators = num_estimators
        self.format_batch_fn = format_batch_fn
        self.optim_recipe = optim_recipe

        self.metric_to_monitor = "val/mean_iou"

        # metrics
        seg_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "mean_iou": MeanIntersectionOverUnion(num_classes=num_classes),
            },
            compute_groups=[["acc", "mean_iou"]],
        )

        self.val_seg_metrics = seg_metrics.clone(prefix="val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="test/")

    def configure_optimizers(self):
        return self.optim_recipe(self.model)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def on_train_start(self) -> None:
        init_metrics = {k: 0 for k in self.val_seg_metrics}
        init_metrics.update({k: 0 for k in self.test_seg_metrics})

        self.logger.log_hyperparams(self.hparams, init_metrics)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        img, target = batch
        img, target = self.format_batch_fn((img, target))
        logits = self.forward(img)
        logits = rearrange(logits, "b c h w -> (b h w) c")
        target = target.flatten()
        valid_mask = target != 255
        loss = self.criterion(logits[valid_mask], target[valid_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        img, target = batch
        logits = self.forward(img)
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
