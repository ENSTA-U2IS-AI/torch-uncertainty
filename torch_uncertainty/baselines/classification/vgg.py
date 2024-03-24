from typing import Literal

from torch import nn

from torch_uncertainty.models import mc_dropout
from torch_uncertainty.models.vgg import (
    packed_vgg11,
    packed_vgg13,
    packed_vgg16,
    packed_vgg19,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
)
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget


class VGGBaseline(ClassificationRoutine):
    single = ["std"]
    ensemble = ["mc-dropout", "packed"]
    versions = {
        "std": [vgg11, vgg13, vgg16, vgg19],
        "mc-dropout": [vgg11, vgg13, vgg16, vgg19],
        "packed": [
            packed_vgg11,
            packed_vgg13,
            packed_vgg16,
            packed_vgg19,
        ],
    }
    archs = [11, 13, 16, 19]

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        version: Literal["std", "mc-dropout", "packed"],
        arch: int,
        style: str = "imagenet",
        num_estimators: int = 1,
        dropout_rate: float = 0.0,
        last_layer_dropout: bool = False,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        groups: int = 1,
        alpha: int | None = None,
        gamma: int = 1,
        ood_criterion: Literal[
            "msp", "logit", "energy", "entropy", "mi", "vr"
        ] = "msp",
        log_plots: bool = False,
        save_in_csv: bool = False,
        calibration_set: Literal["val", "test"] | None = None,
        eval_ood: bool = False,
        eval_grouping_loss: bool = False,
    ) -> None:
        r"""VGG backbone baseline for classification providing support for
        various versions and architectures.

        Args:
            num_classes (int): Number of classes to predict.
            in_channels (int): Number of input channels.
            loss (nn.Module): Training loss.
            version (str):
                Determines which VGG version to use:

                - ``"std"``: original VGG
                - ``"mc-dropout"``: Monte Carlo Dropout VGG
                - ``"packed"``: Packed-Ensembles VGG

            arch (int):
                Determines which VGG architecture to use:

                - ``11``: VGG-11
                - ``13``: VGG-13
                - ``16``: VGG-16
                - ``19``: VGG-19

            style (str, optional): Which VGG style to use. Defaults to
            ``imagenet``.
            num_estimators (int, optional): Number of estimators in the ensemble.
                Only used if :attr:`version` is either ``"packed"``, ``"batched"``
                or ``"masked"`` Defaults to ``None``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.
            mixtype (str, optional): Mixup type. Defaults to ``"erm"``.
            mixmode (str, optional): Mixup mode. Defaults to ``"elem"``.
            dist_sim (str, optional): Distance similarity. Defaults to ``"emb"``.
            kernel_tau_max (float, optional): Maximum value for the kernel tau.
                Defaults to ``1.0``.
            kernel_tau_std (float, optional): Standard deviation for the kernel
                tau. Defaults to ``0.5``.
            mixup_alpha (float, optional): Alpha parameter for Mixup. Defaults
                to ``0``.
            cutmix_alpha (float, optional): Alpha parameter for CutMix.
                Defaults to ``0``.
            last_layer_dropout (bool): whether to apply dropout to the last layer only.
            groups (int, optional): Number of groups in convolutions. Defaults to
                ``1``.
            alpha (float, optional): Expansion factor affecting the width of the
                estimators. Only used if :attr:`version` is ``"packed"``. Defaults
                to ``None``.
            gamma (int, optional): Number of groups within each estimator. Only
                used if :attr:`version` is ``"packed"`` and scales with
                :attr:`groups`. Defaults to ``1s``.
            ood_criterion (str, optional): OOD criterion. Defaults to ``"msp"``.
                MSP is the maximum softmax probability, logit is the maximum
                logit, entropy is the entropy of the mean prediction, mi is the
                mutual information of the ensemble and vr is the variation ratio
                of the ensemble.
            log_plots (bool, optional): Indicates whether to log the plots or not.
                Defaults to ``False``.
            save_in_csv (bool, optional): Indicates whether to save the results in
                a csv file or not. Defaults to ``False``.
            calibration_set (Callable, optional): Calibration set. Defaults to
                ``None``.
            eval_ood (bool, optional): Indicates whether to evaluate the
                OOD detection or not. Defaults to ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.

        Raises:
            ValueError: If :attr:`version` is not either ``"std"``,
                ``"packed"``, ``"batched"`` or ``"masked"``.

        Returns:
            LightningModule: VGG baseline ready for training and evaluation.
        """
        params = {
            "dropout_rate": dropout_rate,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
            "groups": groups,
        }

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        format_batch_fn = nn.Identity()

        if version == "std":
            params |= {
                "dropout_rate": dropout_rate,
            }

        elif version == "mc-dropout":
            params |= {
                "dropout_rate": dropout_rate,
                "num_estimators": num_estimators,
            }

        if version in self.ensemble:
            params |= {
                "num_estimators": num_estimators,
            }

            if version != "mc-dropout":
                format_batch_fn = RepeatTarget(num_repeats=num_estimators)

        if version == "packed":
            params |= {
                "alpha": alpha,
                "style": style,
                "gamma": gamma,
            }

        if version == "mc-dropout":  # std VGGs don't have `num_estimators`
            del params["num_estimators"]
        model = self.versions[version][self.archs.index(arch)](**params)
        if version == "mc-dropout":
            model = mc_dropout(
                model=model,
                num_estimators=num_estimators,
                last_layer=last_layer_dropout,
            )
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            num_estimators=num_estimators,
            format_batch_fn=format_batch_fn,
            mixtype=mixtype,
            mixmode=mixmode,
            dist_sim=dist_sim,
            kernel_tau_max=kernel_tau_max,
            kernel_tau_std=kernel_tau_std,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            eval_ood=eval_ood,
            ood_criterion=ood_criterion,
            log_plots=log_plots,
            save_in_csv=save_in_csv,
            calibration_set=calibration_set,
            eval_grouping_loss=eval_grouping_loss,
        )
        self.save_hyperparameters(ignore=["loss"])
