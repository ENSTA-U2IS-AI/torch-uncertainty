from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.data import Mixup as timm_Mixup
from torch import Tensor, nn
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.metrics import (
    CE,
    FPR95,
    BrierScore,
    Disagreement,
    Entropy,
    GroupingLoss,
    MutualInformation,
    NegativeLogLikelihood,
    VariationRatio,
)
from torch_uncertainty.plotting_utils import plot_hist
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.transforms import Mixup, MixupIO, RegMixup, WarpingMixup
from torch_uncertainty.utils import csv_writer


class ClassificationRoutine(LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: type[nn.Module],
        num_estimators: int,
        format_batch_fn: nn.Module | None = None,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1.0,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        eval_ood: bool = False,
        eval_grouping_loss: bool = False,
        ood_criterion: Literal[
            "msp", "logit", "energy", "entropy", "mi", "vr"
        ] = "msp",
        log_plots: bool = False,
        save_in_csv: bool = False,
        calibration_set: Literal["val", "test"] | None = None,
    ) -> None:
        """Classification routine.

        Args:
            num_classes (int): Number of classes.
            model (nn.Module): Model to train.
            loss (type[nn.Module]): Loss function.
            num_estimators (int): _description_
            optimization_procedure (Any): Optimization procedure.
            format_batch_fn (nn.Module, optional): Function to format the batch.
                Defaults to :class:`torch.nn.Identity()`.
            mixtype (str, optional): Mixup type. Defaults to ``"erm"``.
            mixmode (str, optional): Mixup mode. Defaults to ``"elem"``.
            dist_sim (str, optional): Distance similarity. Defaults to ``"emb"``.
            kernel_tau_max (float, optional): Maximum value for the kernel tau.
                Defaults to 1.0.
            kernel_tau_std (float, optional): Standard deviation for the kernel tau.
                Defaults to 0.5.
            mixup_alpha (float, optional): Alpha parameter for Mixup. Defaults to 0.
            cutmix_alpha (float, optional): Alpha parameter for Cutmix.
                Defaults to 0.
            eval_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance or not. Defaults to ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
            ood_criterion (str, optional): OOD criterion. Defaults to ``"msp"``.
                MSP is the maximum softmax probability, logit is the maximum
                logit, energy the logsumexp of the mean logits, entropy the
                entropy of the mean prediction, mi is the mutual information of
                the ensemble and vr is the variation ratio of the ensemble.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            save_in_csv(bool, optional): __TODO__
            calibration_set (Callable, optional): Function to get the calibration
                set. Defaults to ``None``.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        super().__init__()

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        if not isinstance(num_estimators, int) and num_estimators < 1:
            raise ValueError(
                "The number of estimators must be a positive integer >= 1."
                f"Got {num_estimators}."
            )

        if ood_criterion not in [
            "msp",
            "logit",
            "energy",
            "entropy",
            "mi",
            "vr",
        ]:
            raise ValueError(
                "The OOD criterion must be one of 'msp', 'logit', 'energy', 'entropy',"
                f" 'mi' or 'vr'. Got {ood_criterion}."
            )

        if num_estimators == 1 and ood_criterion in ["mi", "vr"]:
            raise ValueError(
                "You cannot use mutual information or variation ratio with a single"
                " model."
            )

        if num_estimators == 1 and eval_grouping_loss:
            raise NotImplementedError(
                "Groupng loss for ensembles is not yet implemented. Raise an issue if needed."
            )

        if eval_grouping_loss and not hasattr(model, "feats_forward"):
            raise ValueError(
                "Your model must have a `feats_forward` method to compute the "
                "grouping loss."
            )

        if eval_grouping_loss and not (
            hasattr(model, "classification_head") or hasattr(model, "linear")
        ):
            raise ValueError(
                "Your model must have a `classification_head` or `linear` "
                "attribute to compute the grouping loss."
            )

        self.num_classes = num_classes
        self.num_estimators = num_estimators
        self.eval_ood = eval_ood
        self.eval_grouping_loss = eval_grouping_loss
        self.ood_criterion = ood_criterion
        self.log_plots = log_plots
        self.save_in_csv = save_in_csv
        self.calibration_set = calibration_set
        self.binary_cls = num_classes == 1

        self.model = model
        self.loss = loss
        # batch format
        self.format_batch_fn = format_batch_fn

        # metrics
        if self.binary_cls:
            cls_metrics = MetricCollection(
                {
                    "acc": Accuracy(task="binary"),
                    "ece": CE(task="binary"),
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
                    "ece": CE(task="multiclass", num_classes=self.num_classes),
                    "brier": BrierScore(num_classes=self.num_classes),
                },
                compute_groups=False,
            )

        self.val_cls_metrics = cls_metrics.clone(prefix="cls_val/")
        self.test_cls_metrics = cls_metrics.clone(prefix="cls_test/")

        if self.calibration_set is not None:
            self.ts_cls_metrics = cls_metrics.clone(prefix="ts_")

        self.test_entropy_id = Entropy()

        if self.eval_ood:
            ood_metrics = MetricCollection(
                {
                    "fpr95": FPR95(pos_label=1),
                    "auroc": BinaryAUROC(),
                    "aupr": BinaryAveragePrecision(),
                },
                compute_groups=[["auroc", "aupr"], ["fpr95"]],
            )
            self.test_ood_metrics = ood_metrics.clone(prefix="ood/")
            self.test_entropy_ood = Entropy()

        self.mixtype = mixtype
        self.mixmode = mixmode
        self.dist_sim = dist_sim
        if num_estimators == 1:
            if mixup_alpha < 0 or cutmix_alpha < 0:
                raise ValueError(
                    "Cutmix alpha and Mixup alpha must be positive."
                    f"Got {mixup_alpha} and {cutmix_alpha}."
                )

            self.mixup = self.init_mixup(
                mixup_alpha, cutmix_alpha, kernel_tau_max, kernel_tau_std
            )

            if self.eval_grouping_loss:
                grouping_loss = MetricCollection(
                    {"grouping_loss": GroupingLoss()}
                )
                self.val_grouping_loss = grouping_loss.clone(prefix="gpl/val_")
                self.test_grouping_loss = grouping_loss.clone(
                    prefix="gpl/test_"
                )

        # Handle ELBO special cases
        self.is_elbo = (
            isinstance(self.loss, partial) and self.loss.func == ELBOLoss
        )

        # Deep Evidential Classification
        self.is_dec = self.loss == DECLoss or (
            isinstance(self.loss, partial) and self.loss.func == DECLoss
        )

        # metrics for ensembles only
        if self.num_estimators > 1:
            ens_metrics = MetricCollection(
                {
                    "disagreement": Disagreement(),
                    "mi": MutualInformation(),
                    "entropy": Entropy(),
                }
            )

            self.test_id_ens_metrics = ens_metrics.clone(prefix="ood/ens_")

            if self.eval_ood:
                self.test_ood_ens_metrics = ens_metrics.clone(prefix="ood/ens_")

        self.id_logit_storage = None
        self.ood_logit_storage = None

    def init_mixup(
        self,
        mixup_alpha: float,
        cutmix_alpha: float,
        kernel_tau_max: float,
        kernel_tau_std: float,
    ) -> Callable:
        if self.mixtype == "timm":
            return timm_Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                mode=self.mixmode,
                num_classes=self.num_classes,
            )
        if self.mixtype == "mixup":
            return Mixup(
                alpha=mixup_alpha,
                mode=self.mixmode,
                num_classes=self.num_classes,
            )
        if self.mixtype == "mixup_io":
            return MixupIO(
                alpha=mixup_alpha,
                mode=self.mixmode,
                num_classes=self.num_classes,
            )
        if self.mixtype == "regmixup":
            return RegMixup(
                alpha=mixup_alpha,
                mode=self.mixmode,
                num_classes=self.num_classes,
            )
        if self.mixtype == "kernel_warping":
            return WarpingMixup(
                alpha=mixup_alpha,
                mode=self.mixmode,
                num_classes=self.num_classes,
                apply_kernel=True,
                tau_max=kernel_tau_max,
                tau_std=kernel_tau_std,
            )
        return nn.Identity()

    def on_train_start(self) -> None:
        init_metrics = {k: 0 for k in self.val_cls_metrics}
        init_metrics.update({k: 0 for k in self.test_cls_metrics})

        if self.logger is not None:  # coverage: ignore
            self.logger.log_hyperparams(
                self.hparams,
                init_metrics,
            )

    def on_test_start(self) -> None:
        if isinstance(self.calibration_set, str) and self.calibration_set in [
            "val",
            "test",
        ]:
            dataset = (
                self.trainer.datamodule.val_dataloader().dataset
                if self.calibration_set == "val"
                else self.trainer.datamodule.test_dataloader().dataset
            )
            self.scaler = TemperatureScaler(device=self.device).fit(
                model=self.model, calibration_set=dataset
            )
            self.cal_model = torch.nn.Sequential(self.model, self.scaler)
        else:
            self.scaler = None
            self.cal_model = None

        if (
            self.eval_ood
            and self.log_plots
            and isinstance(self.logger, TensorBoardLogger)
        ):
            self.id_logit_storage = []
            self.ood_logit_storage = []

    @property
    def criterion(self) -> nn.Module:
        if self.is_elbo:
            self.loss = partial(self.loss, model=self.model)
        return self.loss()

    def forward(self, inputs: Tensor, save_feats: bool = False) -> Tensor:
        """Forward pass of the model.

        Args:
            inputs (Tensor): Input tensor.
            save_feats (bool, optional): Whether to store the features or
                not. Defaults to ``False``.

        Note:
            The features are stored in the :attr:`features` attribute.
        """
        if save_feats:
            self.features = self.model.feats_forward(inputs)
            if hasattr(self.model, "classification_head"):  # coverage: ignore
                logits = self.model.classification_head(self.features)
            else:
                logits = self.model.linear(self.features)
        else:
            self.features = None
            logits = self.model(inputs)
        return logits

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        # Mixup only for single models
        if self.num_estimators == 1:
            if self.mixtype == "kernel_warping":
                if self.dist_sim == "emb":
                    with torch.no_grad():
                        feats = self.model.feats_forward(batch[0]).detach()

                    batch = self.mixup(*batch, feats)
                elif self.dist_sim == "inp":
                    batch = self.mixup(*batch, batch[0])
            else:
                batch = self.mixup(*batch)

        inputs, targets = self.format_batch_fn(batch)

        if self.is_elbo:
            loss = self.criterion(inputs, targets)
        else:
            logits = self.forward(inputs)
            # BCEWithLogitsLoss expects float targets
            if self.binary_cls and self.loss == nn.BCEWithLogitsLoss:
                logits = logits.squeeze(-1)
                targets = targets.float()

            if not self.is_dec:
                loss = self.criterion(logits, targets)
            else:
                loss = self.criterion(logits, targets, self.current_epoch)

        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs)  # (m*b, c)
        logits = rearrange(logits, "(m b) c -> b m c", m=self.num_estimators)

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits).squeeze(-1)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)
        self.val_cls_metrics.update(probs, targets)

        if self.eval_grouping_loss:
            self.val_grouping_loss.update(probs, targets, self.features)

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        inputs, targets = batch
        logits = self.forward(
            inputs, save_feats=self.eval_grouping_loss
        )  # (m*b, c)
        if logits.size(0) % self.num_estimators != 0:  # coverage: ignore
            raise ValueError(
                f"The number of predicted samples {logits.size(0)} is not "
                "divisible by the reported number of estimators "
                f"{self.num_estimators} of the routine. Please check the "
                "correspondence between these values."
            )
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits).squeeze(-1)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)

        confs = probs.max(-1)[0]

        if self.ood_criterion == "logit":
            ood_scores = -logits.mean(dim=1).max(dim=-1).values
        elif self.ood_criterion == "energy":
            ood_scores = -logits.mean(dim=1).logsumexp(dim=-1)
        elif self.ood_criterion == "entropy":
            ood_scores = (
                torch.special.entr(probs_per_est).sum(dim=-1).mean(dim=1)
            )
        elif self.ood_criterion == "mi":
            mi_metric = MutualInformation(reduction="none")
            ood_scores = mi_metric(probs_per_est)
        elif self.ood_criterion == "vr":
            vr_metric = VariationRatio(reduction="none", probabilistic=False)
            ood_scores = vr_metric(probs_per_est.transpose(0, 1))
        else:
            ood_scores = -confs

        # Scaling for single models
        if (
            self.num_estimators == 1
            and self.calibration_set is not None
            and self.scaler is not None
            and self.cal_model is not None
        ):
            cal_logits = self.cal_model(inputs)
            cal_probs = F.softmax(cal_logits, dim=-1)
            self.ts_cls_metrics.update(cal_probs, targets)

        if dataloader_idx == 0:
            # squeeze if binary classification only for binary metrics
            self.test_cls_metrics.update(
                probs.squeeze(-1) if self.binary_cls else probs,
                targets,
            )
            if self.eval_grouping_loss:
                self.test_grouping_loss.update(probs, targets, self.features)

            self.log_dict(
                self.test_cls_metrics, on_epoch=True, add_dataloader_idx=False
            )
            self.test_entropy_id(probs)
            self.log(
                "cls_test/entropy",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )

            if self.num_estimators > 1:
                self.test_id_ens_metrics.update(probs_per_est)

            if self.eval_ood:
                self.test_ood_metrics.update(
                    ood_scores, torch.zeros_like(targets)
                )

            if self.id_logit_storage is not None:
                self.id_logit_storage.append(logits.detach().cpu())

        elif self.eval_ood and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_scores, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.log(
                "ood/entropy",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            if self.num_estimators > 1:
                self.test_ood_ens_metrics.update(probs_per_est)

            if self.ood_logit_storage is not None:
                self.ood_logit_storage.append(logits.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_cls_metrics.compute())
        self.val_cls_metrics.reset()

        if self.eval_grouping_loss:
            self.log_dict(self.val_grouping_loss.compute())
            self.val_grouping_loss.reset()

    def on_test_epoch_end(self) -> None:
        # already logged
        result_dict = self.test_cls_metrics.compute()

        # already logged
        result_dict.update({"cls_test/entropy": self.test_entropy_id.compute()})

        if (
            self.num_estimators == 1
            and self.calibration_set is not None
            and self.scaler is not None
            and self.cal_model is not None
        ):
            tmp_metrics = self.ts_cls_metrics.compute()
            self.log_dict(tmp_metrics)
            result_dict.update(tmp_metrics)
            self.ts_cls_metrics.reset()

        if self.eval_grouping_loss:
            self.log_dict(
                self.test_grouping_loss.compute(),
            )

        if self.num_estimators > 1:
            tmp_metrics = self.test_id_ens_metrics.compute()
            self.log_dict(tmp_metrics)
            result_dict.update(tmp_metrics)
            self.test_id_ens_metrics.reset()

        if self.eval_ood:
            tmp_metrics = self.test_ood_metrics.compute()
            self.log_dict(tmp_metrics)
            result_dict.update(tmp_metrics)
            self.test_ood_metrics.reset()

            # already logged
            result_dict.update({"ood/entropy": self.test_entropy_ood.compute()})

            if self.num_estimators > 1:
                tmp_metrics = self.test_ood_ens_metrics.compute()
                self.log_dict(tmp_metrics)
                result_dict.update(tmp_metrics)
                self.test_ood_ens_metrics.reset()

        if isinstance(self.logger, TensorBoardLogger) and self.log_plots:
            self.logger.experiment.add_figure(
                "Calibration Plot", self.test_cls_metrics["ece"].plot()[0]
            )

            # plot histograms of logits and likelihoods
            if self.eval_ood:
                id_logits = torch.cat(self.id_logit_storage, dim=0)
                ood_logits = torch.cat(self.ood_logit_storage, dim=0)

                id_probs = F.softmax(id_logits, dim=-1)
                ood_probs = F.softmax(ood_logits, dim=-1)

                logits_fig = plot_hist(
                    [
                        id_logits.mean(1).max(-1).values,
                        ood_logits.mean(1).max(-1).values,
                    ],
                    20,
                    "Histogram of the logits",
                )[0]
                probs_fig = plot_hist(
                    [
                        id_probs.mean(1).max(-1).values,
                        ood_probs.mean(1).max(-1).values,
                    ],
                    20,
                    "Histogram of the likelihoods",
                )[0]
                self.logger.experiment.add_figure("Logit Histogram", logits_fig)
                self.logger.experiment.add_figure(
                    "Likelihood Histogram", probs_fig
                )

        if self.save_in_csv:
            self.save_results_to_csv(result_dict)

    def save_results_to_csv(self, results: dict[str, float]) -> None:
        if self.logger is not None:
            csv_writer(
                Path(self.logger.log_dir) / "results.csv",
                results,
            )
