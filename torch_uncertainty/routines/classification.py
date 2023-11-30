from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from functools import partial
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.memory import get_model_size_mb
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
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
    MutualInformation,
    NegativeLogLikelihood,
    VariationRatio,
)
from torch_uncertainty.plotting_utils import plot_hist
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.transforms import Mixup, MixupIO, RegMixup, WarpingMixup


class ClassificationSingle(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: type[nn.Module],
        optimization_procedure: Any,
        format_batch_fn: nn.Module | None = None,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1.0,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        evaluate_ood: bool = False,
        use_entropy: bool = False,
        use_logits: bool = False,
        log_plots: bool = False,
        calibration_set: Callable | None = None,
        **kwargs,
    ) -> None:
        """Classification routine for single models.

        Args:
            num_classes (int): Number of classes.
            model (nn.Module): Model to train.
            loss (type[nn.Module]): Loss function.
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
            evaluate_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance or not. Defaults to ``False``.
            use_entropy (bool, optional): Indicates whether to use the entropy
                values as the OOD criterion or not. Defaults to ``False``.
            use_logits (bool, optional): Indicates whether to use the logits as the
                OOD criterion or not. Defaults to ``False``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            calibration_set (Callable, optional): Function to get the calibration
                set. Defaults to ``None``.
            kwargs (Any): Additional arguments.

        Note:
            The default OOD criterion is the softmax confidence score.

        Warning:
            Make sure at most only one of :attr:`use_entropy` and :attr:`use_logits`
            attributes is set to ``True``. Otherwise a :class:`ValueError()` will
            be raised.
        """
        super().__init__()

        if format_batch_fn is None:
            format_batch_fn = nn.Identity()

        self.save_hyperparameters(
            ignore=[
                "model",
                "loss",
                "optimization_procedure",
                "format_batch_fn",
                "calibration_set",
            ]
        )

        if (use_logits + use_entropy) > 1:
            raise ValueError("You cannot choose more than one OOD criterion.")

        self.num_classes = num_classes
        self.evaluate_ood = evaluate_ood
        self.use_logits = use_logits
        self.use_entropy = use_entropy
        self.log_plots = log_plots

        self.calibration_set = calibration_set

        self.binary_cls = num_classes == 1

        self.model = model
        self.loss = loss
        self.optimization_procedure = optimization_procedure
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

        self.val_cls_metrics = cls_metrics.clone(prefix="hp/val_")
        self.test_cls_metrics = cls_metrics.clone(prefix="hp/test_")

        if self.calibration_set is not None:
            self.ts_cls_metrics = cls_metrics.clone(prefix="hp/ts_")

        self.test_entropy_id = Entropy()

        if self.evaluate_ood:
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

        self.mixtype = mixtype
        self.mixmode = mixmode
        self.dist_sim = dist_sim

        self.mixup = self.init_mixup(
            mixup_alpha, cutmix_alpha, kernel_tau_max, kernel_tau_std
        )

        # Handle ELBO special cases
        self.is_elbo = (
            isinstance(self.loss, partial) and self.loss.func == ELBOLoss
        )

        # DEC
        self.is_dec = self.loss == DECLoss or (
            isinstance(self.loss, partial) and self.loss.func == DECLoss
        )

    def configure_optimizers(self) -> Any:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        if self.is_elbo:
            self.loss = partial(self.loss, model=self.model)
        return self.loss()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model.forward(inputs)

    def on_train_start(self) -> None:
        # hyperparameters for performances
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:  # coverage: ignore
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
                    "hp/ts_test_nll": 0,
                    "hp/ts_test_ece": 0,
                    "hp/ts_test_brier": 0,
                },
            )

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
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
        logits = self.forward(inputs)

        if self.binary_cls:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)

        self.val_cls_metrics.update(probs, targets)

    def validation_epoch_end(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]
    ) -> None:
        self.log_dict(self.val_cls_metrics.compute())
        self.val_cls_metrics.reset()

    def on_test_start(self) -> None:
        if self.calibration_set is not None:
            self.scaler = TemperatureScaler(device=self.device).fit(
                model=self.model, calibration_set=self.calibration_set()
            )
            self.cal_model = torch.nn.Sequential(self.model, self.scaler)
        else:
            self.scaler = None
            self.cal_model = None

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int | None = 0,
    ) -> Tensor:
        inputs, targets = batch
        logits = self.forward(inputs)

        if self.binary_cls:
            probs = torch.sigmoid(logits).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)

        # self.cal_plot.update(probs, targets)
        confs = probs.max(dim=-1)[0]

        if self.use_logits:
            ood_scores = -logits.max(dim=-1)[0]
        elif self.use_entropy:
            ood_scores = torch.special.entr(probs).sum(dim=-1)
        else:
            ood_scores = -confs

        if (
            self.calibration_set is not None
            and self.scaler is not None
            and self.cal_model is not None
        ):
            cal_logits = self.cal_model(inputs)
            cal_probs = F.softmax(cal_logits, dim=-1)
            self.ts_cls_metrics.update(cal_probs, targets)

        if dataloader_idx == 0:
            self.test_cls_metrics.update(probs, targets)
            self.test_entropy_id(probs)
            self.log(
                "hp/test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            if self.evaluate_ood:
                self.test_ood_metrics.update(
                    ood_scores, torch.zeros_like(targets)
                )
        elif self.evaluate_ood and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_scores, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.log(
                "hp/test_entropy_ood",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        return logits

    def test_epoch_end(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]
    ) -> None:
        self.log_dict(
            self.test_cls_metrics.compute(),
        )

        if (
            self.calibration_set is not None
            and self.scaler is not None
            and self.cal_model is not None
        ):
            self.log_dict(self.ts_cls_metrics.compute())
            self.ts_cls_metrics.reset()

        if self.evaluate_ood:
            self.log_dict(
                self.test_ood_metrics.compute(),
            )
            self.test_ood_metrics.reset()

        if isinstance(self.logger, TensorBoardLogger) and self.log_plots:
            self.logger.experiment.add_figure(
                "Calibration Plot", self.test_cls_metrics["ece"].plot()[0]
            )

            if self.evaluate_ood:
                id_logits = torch.cat(outputs[0], 0).float().cpu()
                ood_logits = torch.cat(outputs[1], 0).float().cpu()

                id_probs = F.softmax(id_logits, dim=-1)
                ood_probs = F.softmax(ood_logits, dim=-1)

                logits_fig = plot_hist(
                    [id_logits.max(-1).values, ood_logits.max(-1).values],
                    20,
                    "Histogram of the logits",
                )[0]
                probs_fig = plot_hist(
                    [id_probs.max(-1).values, ood_probs.max(-1).values],
                    20,
                    "Histogram of the likelihoods",
                )[0]
                self.logger.experiment.add_figure("Logit Histogram", logits_fig)
                self.logger.experiment.add_figure(
                    "Likelihood Histogram", probs_fig
                )

        self.test_cls_metrics.reset()

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
        return lambda x, y: (x, y)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options.

        Args:
            parent_parser (ArgumentParser): Parent parser to be completed.

        Adds:
        - ``--entropy``: sets :attr:`use_entropy` to ``True``.
        - ``--logits``: sets :attr:`use_logits` to ``True``.
        - ``--mixup_alpha``: sets :attr:`mixup_alpha` for Mixup
        - ``--cutmix_alpha``: sets :attr:`cutmix_alpha` for Cutmix
        - ``--mixtype``: sets :attr:`mixtype` for Mixup
        - ``--mixmode``: sets :attr:`mixmode` for Mixup
        - ``--dist_sim``: sets :attr:`dist_sim` for Mixup
        - ``--kernel_tau_max``: sets :attr:`kernel_tau_max` for Mixup
        - ``--kernel_tau_std``: sets :attr:`kernel_tau_std` for Mixup
        """
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )

        # Mixup args
        parent_parser.add_argument(
            "--mixup_alpha", dest="mixup_alpha", type=float, default=0
        )
        parent_parser.add_argument(
            "--cutmix_alpha", dest="cutmix_alpha", type=float, default=0
        )
        parent_parser.add_argument(
            "--mixtype", dest="mixtype", type=str, default="erm"
        )
        parent_parser.add_argument(
            "--mixmode", dest="mixmode", type=str, default="elem"
        )
        parent_parser.add_argument(
            "--dist_sim", dest="dist_sim", type=str, default="emb"
        )
        parent_parser.add_argument(
            "--kernel_tau_max", dest="kernel_tau_max", type=float, default=1.0
        )
        parent_parser.add_argument(
            "--kernel_tau_std", dest="kernel_tau_std", type=float, default=0.5
        )
        return parent_parser


class ClassificationEnsemble(ClassificationSingle):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        loss: type[nn.Module],
        optimization_procedure: Any,
        num_estimators: int,
        format_batch_fn: nn.Module | None = None,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1.0,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        evaluate_ood: bool = False,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        log_plots: bool = False,
        **kwargs,
    ) -> None:
        """Classification routine for ensemble models.

        Args:
            num_classes (int): Number of classes.
            model (nn.Module): Model to train.
            loss (type[nn.Module]): Loss function.
            optimization_procedure (Any): Optimization procedure.
            num_estimators (int): Number of estimators in the ensemble.
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
            evaluate_ood (bool, optional): Indicates whether to evaluate the OOD
                detection performance or not. Defaults to ``False``.
            use_entropy (bool, optional): Indicates whether to use the entropy
                values as the OOD criterion or not. Defaults to ``False``.
            use_logits (bool, optional): Indicates whether to use the logits as the
                OOD criterion or not. Defaults to ``False``.
            use_mi (bool, optional): Indicates whether to use the mutual
                information as the OOD criterion or not. Defaults to ``False``.
            use_variation_ratio (bool, optional): Indicates whether to use the
                variation ratio as the OOD criterion or not. Defaults to ``False``.
            log_plots (bool, optional): Indicates whether to log plots from
                metrics. Defaults to ``False``.
            calibration_set (Callable, optional): Function to get the calibration
                set. Defaults to ``None``.
            kwargs (Any): Additional arguments.

        Note:
            The default OOD criterion is the averaged softmax confidence score.

        Warning:
            Make sure at most only one of :attr:`use_entropy`, :attr:`use_logits`
            , :attr:`use_mi`, and :attr:`use_variation_ratio` attributes is set to
            ``True``. Otherwise a :class:`ValueError()` will be raised.
        """
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            format_batch_fn=format_batch_fn,
            mixtype=mixtype,
            mixmode=mixmode,
            dist_sim=dist_sim,
            kernel_tau_max=kernel_tau_max,
            kernel_tau_std=kernel_tau_std,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            evaluate_ood=evaluate_ood,
            use_entropy=use_entropy,
            use_logits=use_logits,
            **kwargs,
        )

        self.num_estimators = num_estimators

        self.use_mi = use_mi
        self.use_variation_ratio = use_variation_ratio
        self.log_plots = log_plots

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

        if self.evaluate_ood:
            self.test_ood_ens_metrics = ens_metrics.clone(
                prefix="hp/test_ood_ens_"
            )

    def on_train_start(self) -> None:
        param = {}
        param["storage"] = f"{get_model_size_mb(self)} MB"
        if self.logger is not None:  # coverage: ignore
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
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        batch = self.mixup(*batch)
        # eventual input repeat is done in the model
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
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int | None = 0,
    ) -> Tensor:
        inputs, targets = batch
        logits = self.forward(inputs)
        logits = rearrange(logits, "(n b) c -> b n c", n=self.num_estimators)

        if self.binary_cls:
            probs_per_est = torch.sigmoid(logits)
        else:
            probs_per_est = F.softmax(logits, dim=-1)

        probs = probs_per_est.mean(dim=1)
        # self.cal_plot.update(probs, targets)
        confs = probs.max(-1)[0]

        if self.use_logits:
            ood_scores = -logits.mean(dim=1).max(dim=-1)[0]
        elif self.use_entropy:
            ood_scores = (
                torch.special.entr(probs_per_est).sum(dim=-1).mean(dim=1)
            )
        elif self.use_mi:
            mi_metric = MutualInformation(reduction="none")
            ood_scores = mi_metric(probs_per_est)
        elif self.use_variation_ratio:
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
            self.test_entropy_id(probs)

            self.test_id_ens_metrics.update(probs_per_est)
            self.log(
                "hp/test_entropy_id",
                self.test_entropy_id,
                on_epoch=True,
                add_dataloader_idx=False,
            )

            if self.evaluate_ood:
                self.test_ood_metrics.update(
                    ood_scores, torch.zeros_like(targets)
                )
        elif self.evaluate_ood and dataloader_idx == 1:
            self.test_ood_metrics.update(ood_scores, torch.ones_like(targets))
            self.test_entropy_ood(probs)
            self.test_ood_ens_metrics.update(probs_per_est)
            self.log(
                "hp/test_entropy_ood",
                self.test_entropy_ood,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        return logits

    def test_epoch_end(
        self, outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT]
    ) -> None:
        self.log_dict(
            self.test_cls_metrics.compute(),
        )

        self.log_dict(
            self.test_id_ens_metrics.compute(),
        )

        if self.evaluate_ood:
            self.log_dict(
                self.test_ood_metrics.compute(),
            )
            self.log_dict(
                self.test_ood_ens_metrics.compute(),
            )

            self.test_ood_metrics.reset()
            self.test_ood_ens_metrics.reset()

        if isinstance(self.logger, TensorBoardLogger) and self.log_plots:
            self.logger.experiment.add_figure(
                "Calibration Plot", self.test_cls_metrics["ece"].plot()[0]
            )

            if self.evaluate_ood:
                id_logits = torch.cat(outputs[0], 0).float().cpu()
                ood_logits = torch.cat(outputs[1], 0).float().cpu()

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

        self.test_cls_metrics.reset()
        self.test_id_ens_metrics.reset()

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the routine's attributes via command-line options.

        Adds:
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
