from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import MetricCollection

from torch_uncertainty.metrics import MeanIntersectionOverUnion


class SegmentationRoutine(LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int,
        format_batch_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.num_classes = num_classes
        self.model = model
        self.loss = loss

        self.metric_to_monitor = "val/mean_iou"

        # metrics
        seg_metrics = MetricCollection(
            {
                "mean_iou": MeanIntersectionOverUnion(num_classes=num_classes),
            }
        )

        self.val_seg_metrics = seg_metrics.clone(prefix="val/")
        self.test_seg_metrics = seg_metrics.clone(prefix="test/")

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
        pred = self.forward(img)
        loss = self.criterion(pred, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        img, targets = batch
        logits = self.forward(img)
        self.val_seg_metrics.update(logits, targets)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        img, target = batch
        pred = self.forward(img)
        self.test_seg_metrics.update(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_seg_metrics.compute())
        self.val_seg_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_seg_metrics.compute())
        self.test_seg_metrics.reset()
