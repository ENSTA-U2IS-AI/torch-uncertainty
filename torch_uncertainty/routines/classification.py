# fmt: off
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from timm.data import Mixup
from torchmetrics import Accuracy, CalibrationError, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
)

from ..metrics import (
    FPR95,
    BrierScore,
    Disagreement,
    Entropy,
    MutualInformation,
    NegativeLogLikelihood,
    VariationRatio,
)


# fmt:on
class ClassificationSingle(pl.LightningModule):
    """
    Args:
        ood_detection (bool, optional): Indicates whether to evaluate the OOD
            detection performance or not. Defaults to ``False``.
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.

    Note:
        The default OOD criterion is the softmax confidence score.

    Warning:
        Make sure at most only one of :attr:`use_entropy` and :attr:`use_logits`
        attributes is set to ``True``. Otherwise a :class:`ValueError()` will
        be raised.
    """

    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: Type[nn.Module],
        optimization_procedure: Any,
        format_batch_fn: nn.Module = nn.Identity(),
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        ood_detection: bool = False,
        use_entropy: bool = False,
        use_logits: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "optimization_procedure",
                "format_batch_fn",
            ]
        )

        if (use_logits + use_entropy) > 1:
            raise ValueError("You cannot choose more than one OOD criterion.")

        self.num_classes = num_classes
        self.ood_detection = ood_detection
        self.use_logits = use_logits
        self.use_entropy = use_entropy

        self.binary_cls = num_classes == 1

        # model
        self.model = model
        # loss
        self.loss = loss
        # optimization procedure
        self.optimization_procedure = optimization_procedure
        # batch format
        self.format_batch_fn = format_batch_fn

        # metrics
        if self.binary_cls:
            cls_metrics = MetricCollection(
                {
                    "acc": BinaryAccuracy(),
                    "ece": BinaryCalibrationError(),
                    "brier": BrierScore(num_classes=1),
                },
                compute_groups=False,
            )
        else:
            cls_metrics = MetricCollection(
                {
                    "nll": NegativeLogLikelihood(),
                    "acc": Accuracy(
                        task="multiclass", num_classes=self.num_classes
                    ),
                    "ece": CalibrationError(
                        task="multiclass", num_classes=self.num_classes
                    ),
                    "brier": BrierScore(num_classes=self.num_classes),
                },
                compute_groups=False,
            )

        self.val_cls_metrics = cls_metrics.clone(prefix="hp/val_")
        self.test_cls_metrics = cls_metrics.clone(prefix="hp/test_")

        self.test_entropy_id = Entropy()

        if self.ood_detection:
            ood_metrics = MetricCollection(
                {
                    "fpr95": FPR95(pos_label=1),
                    "auroc": BinaryAUROC(),
                    "aupr": BinaryAveragePrecision(),
                },
                compute_groups=[["auroc", "aupr"], ["fpr95"]],
            )
            self.test_ood_metrics = ood_metrics.clone(prefix="hp/test_")
            self.test_entropy_ood = Entropy()

        if mixup_alpha < 0 or cutmix_alpha < 0:
            raise ValueError(
                "Cutmix alpha and Mixup alpha must be positive."
                f"Got {mixup_alpha} and {cutmix_alpha}."
            )
        elif mixup_alpha > 0 or cutmix_alpha > 0:
            self.mixup = Mixup(
                mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha
            )
        else:
            self.mixup = lambda x, y: (x, y)

    def configure_optimizers(self) -> Any:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model.forward(input)

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
                    "hp/test_brier": 0,
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
        batch = self.mixup(*batch)
        inputs, targets = self.format_batch_fn(batch)
        logits = self.forward(inputs)

        # BCEWithLogitsLoss expects float targets
        if self.binary_cls and self.loss == nn.BCEWithLogitsLoss:
            logits = logits.squeeze(-1)
            targets = targets.float()

        loss = self.criterion(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.binary_cls:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)

        self.val_cls_metrics.update(probs, targets)

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log_dict(self.val_cls_metrics.compute())
        self.val_cls_metrics.reset()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.binary_cls:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
        confs = probs.max(dim=-1)[0]

        if self.use_logits:
            ood_values = -logits.max(dim=-1)[0]
        elif self.use_entropy:
            ood_values = torch.special.entr(probs).sum(dim=-1)
        else:
            ood_values = -confs

        if dataloader_idx == 0:
            self.test_cls_metrics.update(probs, targets)
            self.test_entropy_id(probs)
            self.log(
                "hp/test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            if self.ood_detection:
                self.test_ood_metrics.update(
                    ood_values, torch.zeros_like(targets)
                )
        elif self.ood_detection and dataloader_idx == 1:
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
        self.test_cls_metrics.reset()

        if self.ood_detection:
            self.log_dict(
                self.test_ood_metrics.compute(),
            )
            self.test_ood_metrics.reset()

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options:

        - ``--mixup``: sets :attr:`mixup_alpha` for Mixup
        - ``--cutmix``: sets :attr:`cutmix_alpha` for Cutmix
        - ``--entropy``: sets :attr:`use_entropy` to ``True``.
        - ``--logits``: sets :attr:`use_logits` to ``True``.
        """
        parent_parser.add_argument(
            "--mixup", dest="mixup_alpha", type=float, default=0
        )
        parent_parser.add_argument(
            "--cutmix", dest="cutmix_alpha", type=float, default=0
        )
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )
        return parent_parser


class ClassificationEnsemble(ClassificationSingle):
    """
    Args:
        ood_detection (bool, optional): Indicates whether to evaluate the OOD
            detection performance or not. Defaults to ``False``.
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.
        use_mi (bool, optional): Indicates whether to use the mutual
            information as the OOD criterion or not. Defaults to ``False``.
        use_variation_ratio (bool, optional): Indicates whether to use the
            variation ratio as the OOD criterion or not. Defaults to ``False``.

    Note:
        The default OOD criterion is the averaged softmax confidence score.

    Warning:
        Make sure at most only one of :attr:`use_entropy`, :attr:`use_logits`
        , :attr:`use_mi`, and :attr:`use_variation_ratio` attributes is set to
        ``True``. Otherwise a :class:`ValueError()` will be raised.
    """

    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: Type[nn.Module],
        optimization_procedure: Any,
        num_estimators: int,
        format_batch_fn: nn.Module = nn.Identity(),
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        ood_detection: bool = False,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            format_batch_fn=format_batch_fn,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            ood_detection=ood_detection,
            use_entropy=use_entropy,
            use_logits=use_logits,
            **kwargs,
        )

        self.num_estimators = num_estimators

        self.use_mi = use_mi
        self.use_variation_ratio = use_variation_ratio

        if (
            self.use_logits
            + self.use_entropy
            + self.use_mi
            + self.use_variation_ratio
        ) > 1:
            raise ValueError("You cannot choose more than one OOD criterion.")

        # metrics for ensembles only
        ens_metrics = MetricCollection(
            {
                "disagreement": Disagreement(),
                "mi": MutualInformation(),
                "entropy": Entropy(),
            }
        )
        self.test_id_ens_metrics = ens_metrics.clone(prefix="hp/test_id_ens_")

        if self.ood_detection:
            self.test_ood_ens_metrics = ens_metrics.clone(
                prefix="hp/test_ood_ens_"
            )

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
                    "hp/test_brier": 0,
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
        batch = self.mixup(*batch)
        inputs, targets = self.format_batch_fn(batch)

        # eventual input repeat is done in the model
        # targets = targets.repeat(self.num_estimators)

        # Computing logits
        logits = self.forward(inputs)

        # BCEWithLogitsLoss expects float targets
        if self.binary_cls and self.loss == nn.BCEWithLogitsLoss:
            logits = logits.squeeze(-1)
            targets = targets.float()

        loss = self.criterion(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        logits = rearrange(logits, "(m b) c -> b m c", m=self.num_estimators)
        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits).squeeze(-1)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)
        self.val_cls_metrics.update(probs, targets)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)
        confs = probs.max(-1)[0]

        if self.use_logits:
            ood_values = -logits.mean(dim=1).max(dim=-1)[0]
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
            # squeeze if binary classification only for binary metrics
            self.test_cls_metrics.update(
                probs.squeeze(-1) if self.binary_cls else probs,
                targets,
            )
            self.test_entropy_id(probs)

            self.test_id_ens_metrics.update(probs_per_est)
            self.log(
                "hp/test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )

            if self.ood_detection:
                self.test_ood_metrics.update(
                    ood_values, torch.zeros_like(targets)
                )
        elif self.ood_detection and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_values, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.test_ood_ens_metrics.update(probs_per_est)
            self.log(
                "hp/test_entropy_ood",
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
        self.test_id_ens_metrics.reset()

        if self.ood_detection:
            self.log_dict(
                self.test_ood_ens_metrics.compute(),
            )
            self.test_ood_ens_metrics.reset()

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options:

        - ``--entropy``: sets :attr:`use_entropy` to ``True``.
        - ``--logits``: sets :attr:`use_logits` to ``True``.
        - ``--mutual_information``: sets :attr:`use_mi` to ``True``.
        - ``--variation_ratio``: sets :attr:`use_variation_ratio` to ``True``.
        - ``--num_estimators``: sets :attr:`num_estimators`.
        """
        parent_parser = ClassificationSingle.add_model_specific_args(
            parent_parser
        )
        # FIXME: should be a str to choose among the available OOD criteria
        # rather than a boolean, but it is not possible since
        # ClassificationSingle and ClassificationEnsemble have different OOD
        # criteria.
        parent_parser.add_argument(
            "--mutual_information",
            dest="use_mi",
            action="store_true",
            default=False,
        )
        parent_parser.add_argument(
            "--variation_ratio",
            dest="use_variation_ratio",
            action="store_true",
            default=False,
        )
        parent_parser.add_argument(
            "--num_estimators",
            type=int,
            default=None,
            help="Number of estimators for ensemble",
        )
        return parent_parser
