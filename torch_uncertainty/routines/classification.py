from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from timm.data import Mixup as timm_Mixup
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
)

from torch_uncertainty.layers import Identity
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.metrics import (
    AUGRC,
    AURC,
    FPR95,
    BrierScore,
    CalibrationError,
    CategoricalNLL,
    CovAt5Risk,
    Disagreement,
    Entropy,
    GroupingLoss,
    MutualInformation,
    RiskAt80Cov,
    VariationRatio,
)
from torch_uncertainty.models import (
    EPOCH_UPDATE_MODEL,
    STEP_UPDATE_MODEL,
)
from torch_uncertainty.post_processing import LaplaceApprox, PostProcessing
from torch_uncertainty.transforms import (
    Mixup,
    MixupIO,
    RegMixup,
    RepeatTarget,
    WarpingMixup,
)
from torch_uncertainty.utils import csv_writer, plot_hist

MIXUP_PARAMS = {
    "mixtype": "erm",
    "mixmode": "elem",
    "dist_sim": "emb",
    "kernel_tau_max": 1.0,
    "kernel_tau_std": 0.5,
    "mixup_alpha": 0,
    "cutmix_alpha": 0,
}


class ClassificationRoutine(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module,
        is_ensemble: bool = False,
        format_batch_fn: nn.Module | None = None,
        optim_recipe: dict | Optimizer | None = None,
        mixup_params: dict | None = None,
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        ood_criterion: Literal[
            "msp", "logit", "energy", "entropy", "mi", "vr"
        ] = "msp",
        post_processing: PostProcessing | None = None,
        calibration_set: Literal["val", "test"] = "val",
        num_calibration_bins: int = 15,
        log_plots: bool = False,
        save_in_csv: bool = False,
    ) -> None:
        r"""Routine for training & testing on **classification** tasks.

        Args:
            model (torch.nn.Module): Model to train.
            num_classes (int): Number of classes.
            loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
            is_ensemble (bool, optional): Indicates whether the model is an
                ensemble at test time or not. Defaults to ``False``.
            format_batch_fn (torch.nn.Module, optional): Function to format the batch.
                Defaults to :class:`torch.nn.Identity()`.
            optim_recipe (dict or torch.optim.Optimizer, optional): The optimizer and
                optionally the scheduler to use. Defaults to ``None``.
            mixup_params (dict, optional): Mixup parameters. Can include mixup type,
                mixup mode, distance similarity, kernel tau max, kernel tau std,
                mixup alpha, and cutmix alpha. If None, no mixup augmentations.
                Defaults to ``None``.
            eval_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance. Defaults to ``False``.
            eval_shift (bool, optional): Indicates whether to evaluate the Distribution
                shift performance. Defaults to ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
            ood_criterion (str, optional): OOD criterion. Available options are
                - ``"msp"`` (default): Maximum softmax probability.
                - ``"logit"``: Maximum logit.
                - ``"energy"``: Logsumexp of the mean logits.
                - ``"entropy"``: Entropy of the mean prediction.
                - ``"mi"``: Mutual information of the ensemble.
                - ``"vr"``: Variation ratio of the ensemble.
            post_processing (PostProcessing, optional): Post-processing method
                to train on the calibration set. No post-processing if None.
                Defaults to ``None``.
            calibration_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            num_calibration_bins (int, optional): Number of bins to compute calibration
                metrics. Defaults to ``15``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            save_in_csv(bool, optional): Save the results in csv. Defaults to
                ``False``.

        Warning:
            You must define :attr:`optim_recipe` if you do not use the Lightning CLI.

        Note:
            :attr:`optim_recipe` can be anything that can be returned by
            :meth:`LightningModule.configure_optimizers()`. Find more details
            `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
        """
        super().__init__()
        _classification_routine_checks(
            model=model,
            num_classes=num_classes,
            is_ensemble=is_ensemble,
            ood_criterion=ood_criterion,
            eval_grouping_loss=eval_grouping_loss,
            num_calibration_bins=num_calibration_bins,
            mixup_params=mixup_params,
            post_processing=post_processing,
            format_batch_fn=format_batch_fn,
        )

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.num_classes = num_classes
        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_grouping_loss = eval_grouping_loss
        self.ood_criterion = ood_criterion
        self.log_plots = log_plots
        self.save_in_csv = save_in_csv
        self.calibration_set = calibration_set
        self.binary_cls = num_classes == 1
        self.needs_epoch_update = isinstance(model, EPOCH_UPDATE_MODEL)
        self.needs_step_update = isinstance(model, STEP_UPDATE_MODEL)
        self.num_calibration_bins = num_calibration_bins
        self.model = model
        self.loss = loss
        self.format_batch_fn = format_batch_fn
        self.optim_recipe = optim_recipe
        self.is_ensemble = is_ensemble

        self.post_processing = post_processing
        if self.post_processing is not None:
            self.post_processing.set_model(self.model)

        self._init_metrics()
        self.mixup = self._init_mixup(mixup_params)

        self.is_elbo = isinstance(self.loss, ELBOLoss)
        if self.is_elbo:
            self.loss.set_model(self.model)
        self.is_dec = isinstance(self.loss, DECLoss)

        self.id_logit_storage = None
        self.ood_logit_storage = None

    def _init_metrics(self) -> None:
        task = "binary" if self.binary_cls else "multiclass"

        metrics_dict = {
            "cls/Acc": Accuracy(task=task, num_classes=self.num_classes),
            "cls/Brier": BrierScore(num_classes=self.num_classes),
            "cls/NLL": CategoricalNLL(),
            "cal/ECE": CalibrationError(
                task=task,
                num_bins=self.num_calibration_bins,
                num_classes=self.num_classes,
            ),
            "cal/aECE": CalibrationError(
                task=task,
                adaptive=True,
                num_bins=self.num_calibration_bins,
                num_classes=self.num_classes,
            ),
            "sc/AURC": AURC(),
            "sc/AUGRC": AUGRC(),
            "sc/Cov@5Risk": CovAt5Risk(),
            "sc/Risk@80Cov": RiskAt80Cov(),
        }
        groups = [
            ["cls/Acc"],
            ["cls/Brier"],
            ["cls/NLL"],
            ["cal/ECE", "cal/aECE"],
            ["sc/AURC", "sc/AUGRC", "sc/Cov@5Risk", "sc/Risk@80Cov"],
        ]

        if self.binary_cls:
            metrics_dict |= {
                "cls/AUROC": BinaryAUROC(),
                "cls/AUPR": BinaryAveragePrecision(),
                "cls/FRP95": FPR95(pos_label=1),
            }
            groups.extend([["cls/AUROC", "cls/AUPR"], ["cls/FRP95"]])

        cls_metrics = MetricCollection(metrics_dict, compute_groups=groups)
        self.val_cls_metrics = cls_metrics.clone(prefix="val/")
        self.test_cls_metrics = cls_metrics.clone(prefix="test/")

        if self.post_processing is not None:
            self.post_cls_metrics = cls_metrics.clone(prefix="test/post/")

        self.test_id_entropy = Entropy()

        if self.eval_ood:
            ood_metrics = MetricCollection(
                {
                    "AUROC": BinaryAUROC(),
                    "AUPR": BinaryAveragePrecision(),
                    "FPR95": FPR95(pos_label=1),
                },
                compute_groups=[["AUROC", "AUPR"], ["FPR95"]],
            )
            self.test_ood_metrics = ood_metrics.clone(prefix="ood/")
            self.test_ood_entropy = Entropy()

        if self.eval_shift:
            self.test_shift_metrics = cls_metrics.clone(prefix="shift/")

        # metrics for ensembles only
        if self.is_ensemble:
            ens_metrics = MetricCollection(
                {
                    "Disagreement": Disagreement(),
                    "MI": MutualInformation(),
                    "Entropy": Entropy(),
                }
            )

            self.test_id_ens_metrics = ens_metrics.clone(prefix="test/ens_")

            if self.eval_ood:
                self.test_ood_ens_metrics = ens_metrics.clone(prefix="ood/ens_")

            if self.eval_shift:
                self.test_shift_ens_metrics = ens_metrics.clone(
                    prefix="shift/ens_"
                )

        if self.eval_grouping_loss:
            grouping_loss = MetricCollection(
                {"cls/grouping_loss": GroupingLoss()}
            )
            self.val_grouping_loss = grouping_loss.clone(prefix="val/")
            self.test_grouping_loss = grouping_loss.clone(prefix="test/")

    def _init_mixup(self, mixup_params: dict | None) -> Callable:
        if mixup_params is None:
            mixup_params = {}
        mixup_params = MIXUP_PARAMS | mixup_params
        self.mixup_params = mixup_params

        if mixup_params["mixup_alpha"] < 0 or mixup_params["cutmix_alpha"] < 0:
            raise ValueError(
                "Cutmix alpha and Mixup alpha must be positive."
                f"Got {mixup_params['mixup_alpha']} and {mixup_params['cutmix_alpha']}."
            )

        if mixup_params["mixtype"] == "timm":
            return timm_Mixup(
                mixup_alpha=mixup_params["mixup_alpha"],
                cutmix_alpha=mixup_params["cutmix_alpha"],
                mode=mixup_params["mixmode"],
                num_classes=self.num_classes,
            )
        if mixup_params["mixtype"] == "mixup":
            return Mixup(
                alpha=mixup_params["mixup_alpha"],
                mode=mixup_params["mixmode"],
                num_classes=self.num_classes,
            )
        if mixup_params["mixtype"] == "mixup_io":
            return MixupIO(
                alpha=mixup_params["mixup_alpha"],
                mode=mixup_params["mixmode"],
                num_classes=self.num_classes,
            )
        if mixup_params["mixtype"] == "regmixup":
            return RegMixup(
                alpha=mixup_params["mixup_alpha"],
                mode=mixup_params["mixmode"],
                num_classes=self.num_classes,
            )
        if mixup_params["mixtype"] == "kernel_warping":
            return WarpingMixup(
                alpha=mixup_params["mixup_alpha"],
                mode=mixup_params["mixmode"],
                num_classes=self.num_classes,
                apply_kernel=True,
                tau_max=mixup_params["kernel_tau_max"],
                tau_std=mixup_params["kernel_tau_std"],
            )
        return Identity()

    def _apply_mixup(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        if not self.is_ensemble:
            if self.mixup_params["mixtype"] == "kernel_warping":
                if self.mixup_params["dist_sim"] == "emb":
                    with torch.no_grad():
                        feats = self.model.feats_forward(batch[0]).detach()
                    batch = self.mixup(*batch, feats)
                else:  # self.mixup_params["dist_sim"] == "inp":
                    batch = self.mixup(*batch, batch[0])
            else:
                batch = self.mixup(*batch)
        return batch

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
        if self.post_processing is not None:
            calibration_dataset = (
                self.trainer.datamodule.val_dataloader().dataset
                if self.calibration_set == "val"
                else self.trainer.datamodule.test_dataloader()[0].dataset
            )
            with torch.inference_mode(False):
                self.post_processing.fit(calibration_dataset)

        if self.eval_ood and self.log_plots and isinstance(self.logger, Logger):
            self.id_logit_storage = []
            self.ood_logit_storage = []

        if hasattr(self.model, "need_bn_update"):
            self.model.bn_update(
                self.trainer.train_dataloader, device=self.device
            )

    def forward(self, inputs: Tensor, save_feats: bool = False) -> Tensor:
        """Forward pass of the model.

        Args:
            inputs (Tensor): Input tensor.
            save_feats (bool, optional): Whether to store the features or
                not. Defaults to ``False``.

        Note:
            The features are stored in the :attr:`self.features` attribute.
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
        batch = self._apply_mixup(batch)
        inputs, target = self.format_batch_fn(batch)

        if self.is_elbo:
            loss = self.loss(inputs, target)
        else:
            logits = self.forward(inputs)
            # BCEWithLogitsLoss expects float target
            if self.binary_cls and isinstance(self.loss, nn.BCEWithLogitsLoss):
                logits = logits.squeeze(-1)
                target = target.float()

            if not self.is_dec:
                loss = self.loss(logits, target)
            else:
                loss = self.loss(logits, target, self.current_epoch)
        if self.needs_step_update:
            self.model.update_wrapper(self.current_epoch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        inputs, targets = batch
        logits = self.forward(inputs, save_feats=self.eval_grouping_loss)
        logits = rearrange(logits, "(m b) c -> b m c", b=targets.size(0))

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
        logits = self.forward(inputs, save_feats=self.eval_grouping_loss)
        logits = rearrange(logits, "(n b) c -> b n c", b=targets.size(0))

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits)
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
            self.test_id_entropy(probs)
            self.log(
                "test/cls/Entropy",
                self.test_id_entropy,
                on_epoch=True,
                add_dataloader_idx=False,
            )

            if self.is_ensemble:
                self.test_id_ens_metrics.update(probs_per_est)

            if self.eval_ood:
                self.test_ood_metrics.update(
                    ood_scores, torch.zeros_like(targets)
                )

            if self.id_logit_storage is not None:
                self.id_logit_storage.append(logits.detach().cpu())

            if self.post_processing is not None:
                pp_logits = self.post_processing(inputs)
                if not isinstance(self.post_processing, LaplaceApprox):
                    pp_probs = F.softmax(pp_logits, dim=-1)
                else:
                    pp_probs = pp_logits
                self.post_cls_metrics.update(pp_probs, targets)

        if self.eval_ood and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_scores, torch.ones_like(targets))
            self.test_ood_entropy(probs)
            self.log(
                "ood/Entropy",
                self.test_ood_entropy,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            if self.is_ensemble:
                self.test_ood_ens_metrics.update(probs_per_est)

            if self.ood_logit_storage is not None:
                self.ood_logit_storage.append(logits.detach().cpu())

        if self.eval_shift and dataloader_idx == (2 if self.eval_ood else 1):
            self.test_shift_metrics.update(probs, targets)
            if self.is_ensemble:
                self.test_shift_ens_metrics.update(probs_per_est)

    def on_validation_epoch_end(self) -> None:
        res_dict = self.val_cls_metrics.compute()
        self.log_dict(res_dict, logger=True, sync_dist=True)
        self.log(
            "Acc%",
            res_dict["val/cls/Acc"] * 100,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.val_cls_metrics.reset()

        if self.eval_grouping_loss:
            self.log_dict(self.val_grouping_loss.compute(), sync_dist=True)
            self.val_grouping_loss.reset()

    def on_test_epoch_end(self) -> None:
        # already logged
        result_dict = self.test_cls_metrics.compute()

        # already logged
        result_dict.update(
            {"test/Entropy": self.test_id_entropy.compute()}, sync_dist=True
        )

        if self.post_processing is not None:
            tmp_metrics = self.post_cls_metrics.compute()
            self.log_dict(tmp_metrics, sync_dist=True)
            result_dict.update(tmp_metrics)

        if self.eval_grouping_loss:
            self.log_dict(
                self.test_grouping_loss.compute(),
                sync_dist=True,
            )

        if self.is_ensemble:
            tmp_metrics = self.test_id_ens_metrics.compute()
            self.log_dict(tmp_metrics, sync_dist=True)
            result_dict.update(tmp_metrics)

        if self.eval_ood:
            tmp_metrics = self.test_ood_metrics.compute()
            self.log_dict(tmp_metrics, sync_dist=True)
            result_dict.update(tmp_metrics)

            # already logged
            result_dict.update({"ood/Entropy": self.test_ood_entropy.compute()})

            if self.is_ensemble:
                tmp_metrics = self.test_ood_ens_metrics.compute()
                self.log_dict(tmp_metrics, sync_dist=True)
                result_dict.update(tmp_metrics)

        if self.eval_shift:
            tmp_metrics = self.test_shift_metrics.compute()
            shift_severity = self.trainer.test_dataloaders[
                2 if self.eval_ood else 1
            ].dataset.shift_severity
            tmp_metrics["shift/shift_severity"] = shift_severity
            self.log_dict(tmp_metrics, sync_dist=True)
            result_dict.update(tmp_metrics)

            if self.is_ensemble:
                tmp_metrics = self.test_shift_ens_metrics.compute()
                self.log_dict(tmp_metrics, sync_dist=True)
                result_dict.update(tmp_metrics)

        if isinstance(self.logger, Logger) and self.log_plots:
            self.logger.experiment.add_figure(
                "Reliabity diagram", self.test_cls_metrics["cal/ECE"].plot()[0]
            )
            self.logger.experiment.add_figure(
                "Risk-Coverage curve",
                self.test_cls_metrics["sc/AURC"].plot()[0],
            )
            self.logger.experiment.add_figure(
                "Generalized Risk-Coverage curve",
                self.test_cls_metrics["sc/AUGRC"].plot()[0],
            )

            if self.post_processing is not None:
                self.logger.experiment.add_figure(
                    "Reliabity diagram after calibration",
                    self.post_cls_metrics["cal/ECE"].plot()[0],
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


def _classification_routine_checks(
    model: nn.Module,
    num_classes: int,
    is_ensemble: bool,
    ood_criterion: str,
    eval_grouping_loss: bool,
    num_calibration_bins: int,
    mixup_params: dict | None,
    post_processing: PostProcessing | None,
    format_batch_fn: nn.Module | None,
) -> None:
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

    if not is_ensemble and ood_criterion in ["mi", "vr"]:
        raise ValueError(
            "You cannot use mutual information or variation ratio with a single"
            " model."
        )

    if is_ensemble and eval_grouping_loss:
        raise NotImplementedError(
            "Groupng loss for ensembles is not yet implemented. Raise an issue if needed."
        )

    if num_classes < 1:
        raise ValueError(
            "The number of classes must be a positive integer >= 1."
            f"Got {num_classes}."
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

    if num_calibration_bins < 2:
        raise ValueError(
            f"num_calibration_bins must be at least 2, got {num_calibration_bins}."
        )

    if mixup_params is not None and isinstance(format_batch_fn, RepeatTarget):
        raise ValueError(
            "Mixup is not supported for ensembles at training time. Please set mixup_params to None."
        )

    if post_processing is not None and is_ensemble:
        raise ValueError(
            "Ensembles and post-processing methods cannot be used together. Raise an issue if needed."
        )
