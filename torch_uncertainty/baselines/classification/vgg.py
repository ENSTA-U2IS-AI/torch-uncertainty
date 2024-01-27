from typing import Literal

from torch import nn
from torch.nn.modules import Module

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


class VGG(ClassificationRoutine):
    single = ["vanilla"]
    ensemble = ["mc-dropout", "packed"]
    versions = {
        "vanilla": [vgg11, vgg13, vgg16, vgg19],
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
        loss: type[Module],
        version: Literal["vanilla", "mc-dropout", "packed"],
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
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        log_plots: bool = False,
        save_in_csv: bool = False,
        calibration_set: Literal["val", "test"] | None = None,
        evaluate_ood: bool = False,
    ) -> None:
        r"""VGG backbone baseline for classification providing support for
        various versions and architectures.

        Args:
            num_classes (int): Number of classes to predict.
            in_channels (int): Number of input channels.
            loss (nn.Module): Training loss.
            version (str):
                Determines which VGG version to use:

                - ``"vanilla"``: original VGG
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
            last_layer_dropout (bool, optional): Indicates whether to apply dropout
                to the last layer or not. Defaults to ``False``.
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
            groups (int, optional): Number of groups in convolutions. Defaults to
                ``1``.
            alpha (float, optional): Expansion factor affecting the width of the
                estimators. Only used if :attr:`version` is ``"packed"``. Defaults
                to ``None``.
            gamma (int, optional): Number of groups within each estimator. Only
                used if :attr:`version` is ``"packed"`` and scales with
                :attr:`groups`. Defaults to ``1s``.
            use_entropy (bool, optional): Indicates whether to use the entropy
                values as the OOD criterion or not. Defaults to ``False``.
            use_logits (bool, optional): Indicates whether to use the logits as the
                OOD criterion or not. Defaults to ``False``.
            use_mi (bool, optional): Indicates whether to use the mutual
                information as the OOD criterion or not. Defaults to ``False``.
            use_variation_ratio (bool, optional): Indicates whether to use the
                variation ratio as the OOD criterion or not. Defaults to ``False``.
            log_plots (bool, optional): Indicates whether to log the plots or not.
                Defaults to ``False``.
            save_in_csv (bool, optional): Indicates whether to save the results in
                a csv file or not. Defaults to ``False``.
            calibration_set (Callable, optional): Calibration set. Defaults to
                ``None``.
            evaluate_ood (bool, optional): Indicates whether to evaluate the
                OOD detection or not. Defaults to ``False``.

        Raises:
            ValueError: If :attr:`version` is not either ``"vanilla"``,
                ``"packed"``, ``"batched"`` or ``"masked"``.

        Returns:
            LightningModule: VGG baseline ready for training and evaluation.
        """
        params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
            "groups": groups,
        }

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        format_batch_fn = nn.Identity()

        if version == "vanilla":
            params.update(
                {
                    "dropout_rate": dropout_rate,
                }
            )
        elif version == "mc-dropout":
            params.update(
                {
                    "dropout_rate": dropout_rate,
                    "num_estimators": num_estimators,
                    "last_layer_dropout": last_layer_dropout,
                }
            )
        elif version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "style": style,
                    "gamma": gamma,
                }
            )
            format_batch_fn = RepeatTarget(num_repeats=num_estimators)

        model = self.versions[version][self.archs.index(arch)](**params)
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
            evaluate_ood=evaluate_ood,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
            log_plots=log_plots,
            save_in_csv=save_in_csv,
            calibration_set=calibration_set,
        )
