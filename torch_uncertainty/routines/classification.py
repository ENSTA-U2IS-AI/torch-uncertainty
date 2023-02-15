# fmt: off
from argparse import Namespace
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CalibrationError,
    MetricCollection,
)

from ..metrics import (
    FPR95,
    Disagreement,
    Entropy,
    MutualInformation,
    NegativeLogLikelihood,
    VariationRatio,
)


# fmt:on
class ClassificationSingle(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        use_entropy: bool = False,
        use_logits: bool = False,
    ) -> None:
        super().__init__()

        # FIXME: use ValueError instead
        assert (
            use_logits + use_entropy
        ) <= 1, "You cannot choose more than one OOD criterion."

        self.num_classes = num_classes
        self.use_logits = use_logits
        self.use_entropy = use_entropy

        # metrics
        cls_metrics = MetricCollection(
            {
                "nll": NegativeLogLikelihood(),
                "acc": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ),
                "ece": CalibrationError(
                    task="multiclass", num_classes=self.num_classes
                ),
            },
            compute_groups=False,
        )

        ood_metrics = MetricCollection(
            {
                "fpr95": FPR95(pos_label=1),
                "auroc": AUROC(task="binary"),
                "aupr": AveragePrecision(task="binary"),
            },
            compute_groups=[["auroc", "aupr"], ["fpr95"]],
        )

        self.val_metrics = cls_metrics.clone(prefix="hp/val_")
        self.test_cls_metrics = cls_metrics.clone(prefix="hp/test_")
        self.test_ood_metrics = ood_metrics.clone(prefix="hp/test_")

        self.test_entropy_id = Entropy()
        self.test_entropy_ood = Entropy()

    @property
    def criterion(self) -> nn.Module:
        raise NotImplementedError()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:
            self.logger.log_hyperparams(
                Namespace(**param),
                {
                    "hp/val_nll": 0,
                    "hp/val_acc": 0,
                    "hp/test_acc": 0,
                    "hp/test_nll": 0,
                    "hp/test_ece": 0,
                    "hp/test_entropy_id": 0,
                    "hp/test_entropy_ood": 0,
                    "hp/test_aupr": 0,
                    "hp/test_auroc": 0,
                    "hp/test_fpr95": 0,
                },
            )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        probs = F.softmax(logits, dim=-1)
        self.val_metrics.update(probs, targets)

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        probs = F.softmax(logits, dim=-1)
        confs, _ = probs.max(dim=-1)

        if self.use_logits:
            ood_values, _ = -logits.max(dim=-1)
        elif self.use_entropy:
            ood_values = torch.special.entr(probs).sum(dim=-1)
        else:
            ood_values = -confs

        if dataloader_idx == 0:
            self.test_cls_metrics.update(probs, targets)
            self.test_ood_metrics.update(ood_values, torch.zeros_like(targets))
            self.test_entropy_id(probs)
            self.log(
                "hp/test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        else:
            self.test_ood_metrics.update(ood_values, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.log(
                "hp/test_entropy_ood",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def test_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(
            self.test_cls_metrics.compute(),
        )
        self.log_dict(
            self.test_ood_metrics.compute(),
        )
        self.test_cls_metrics.reset()
        self.test_ood_metrics.reset()


class ClassificationEnsemble(ClassificationSingle):
    def __init__(
        self,
        num_classes: int,
        num_estimators: int,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_entropy=use_entropy,
            use_logits=use_logits,
        )

        self.num_estimators = num_estimators

        self.use_mi = use_mi
        self.use_variation_ratio = use_variation_ratio

        # FIXME: use ValueError instead
        assert (
            self.use_logits
            + self.use_entropy
            + self.use_mi
            + self.use_variation_ratio
        ) <= 1, "You cannot choose more than one OOD criterion."

        # metrics for ensembles only
        ens_metrics = MetricCollection(
            {
                "disagreement": Disagreement(),
                "mi": MutualInformation(),
                "entropy": Entropy(),
            }
        )
        self.test_id_ens_metrics = ens_metrics.clone(prefix="hp/test_id_ens_")
        self.test_ood_ens_metrics = ens_metrics.clone(prefix="hp/test_ood_ens_")

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:
            self.logger.log_hyperparams(
                Namespace(**param),
                {
                    "hp/val_nll": 0,
                    "hp/val_acc": 0,
                    "hp/test_acc": 0,
                    "hp/test_nll": 0,
                    "hp/test_ece": 0,
                    "hp/test_entropy_id": 0,
                    "hp/test_entropy_ood": 0,
                    "hp/test_aupr": 0,
                    "hp/test_auroc": 0,
                    "hp/test_fpr95": 0,
                    "hp/test_id_ens_disagreement": 0,
                    "hp/test_id_ens_mi": 0,
                    "hp/test_id_ens_entropy": 0,
                    "hp/test_ood_ens_disagreement": 0,
                    "hp/test_ood_ens_mi": 0,
                    "hp/test_ood_ens_entropy": 0,
                },
            )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        inputs, targets = batch
        targets = targets.repeat(self.num_estimators)
        return super().training_step((inputs, targets), batch_idx)

    def validation_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        # logits = logits.reshape(self.num_estimators, -1, logits.size(-1))
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)
        probs_per_est = F.softmax(logits, dim=-1)
        probs = probs_per_est.mean(dim=1)
        self.val_metrics.update(probs, targets)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        # logits = logits.reshape(self.num_estimators, -1, logits.size(-1))
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)
        probs_per_est = F.softmax(logits, dim=-1)
        probs = probs_per_est.mean(dim=1)
        confs, _ = probs.max(-1)

        if self.use_logits:
            ood_values, _ = -logits.mean(dim=1).max(dim=-1)
        elif self.use_entropy:
            ood_values = torch.special.entr(probs).sum(dim=-1).mean(dim=1)
        elif self.use_mi:
            mi_metric = MutualInformation(reduction="none")
            ood_values = mi_metric(probs_per_est)
        elif self.use_variation_ratio:
            vr_metric = VariationRatio(reduction="none", probabilistic=False)
            ood_values = vr_metric(probs_per_est.transpose(0, 1))
        else:
            ood_values = -confs

        if dataloader_idx == 0:
            self.test_cls_metrics.update(probs, targets)
            self.test_ood_metrics.update(ood_values, torch.zeros_like(targets))
            self.test_entropy_id(probs)
            self.test_id_ens_metrics.update(probs_per_est)
            self.log(
                "test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        else:
            self.test_ood_metrics.update(ood_values, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.test_ood_ens_metrics.update(probs_per_est)
            self.log(
                "test_entropy_ood",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def test_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        super().test_epoch_end(outputs)
        self.log_dict(
            self.test_id_ens_metrics.compute(),
        )
        self.log_dict(
            self.test_ood_ens_metrics.compute(),
        )
        self.test_id_ens_metrics.reset()
        self.test_ood_ens_metrics.reset()
