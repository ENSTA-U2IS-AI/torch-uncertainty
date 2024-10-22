from typing import Literal

from torch import nn
from torch.optim import Optimizer

from torch_uncertainty.models import mc_dropout
from torch_uncertainty.models.resnet import (
    batched_resnet,
    lpbnn_resnet,
    masked_resnet,
    mimo_resnet,
    packed_resnet,
    resnet,
)
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.transforms import MIMOBatchFormat, RepeatTarget

ENSEMBLE_METHODS = [
    "packed",
    "batched",
    "lpbnn",
    "masked",
    "mc-dropout",
    "mimo",
]


class ResNetBaseline(ClassificationRoutine):
    versions = {
        "std": resnet,
        "packed": packed_resnet,
        "batched": batched_resnet,
        "lpbnn": lpbnn_resnet,
        "masked": masked_resnet,
        "mimo": mimo_resnet,
        "mc-dropout": resnet,
    }
    archs = [18, 20, 34, 44, 50, 56, 101, 110, 152, 1202]

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        version: Literal[
            "std",
            "mc-dropout",
            "packed",
            "batched",
            "lpbnn",
            "masked",
            "mimo",
        ],
        arch: int,
        style: str = "imagenet",
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
        num_estimators: int = 1,
        dropout_rate: float = 0.0,
        optim_recipe: dict | Optimizer | None = None,
        mixup_params: dict | None = None,
        last_layer_dropout: bool = False,
        width_multiplier: float = 1.0,
        groups: int = 1,
        scale: float | None = None,
        alpha: int | None = None,
        gamma: int = 1,
        rho: float = 1.0,
        batch_repeat: int = 1,
        ood_criterion: Literal[
            "msp", "logit", "energy", "entropy", "mi", "vr"
        ] = "msp",
        log_plots: bool = False,
        save_in_csv: bool = False,
        calibration_set: Literal["val", "test"] = "val",
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        num_calibration_bins: int = 15,
        pretrained: bool = False,
    ) -> None:
        r"""ResNet backbone baseline for classification providing support for
        various versions and architectures.

        Args:
            num_classes (int): Number of classes to predict.
            in_channels (int): Number of input channels.
            loss (nn.Module): Training loss.
            optim_recipe (Any): optimization recipe, corresponds to
                what expect the `LightningModule.configure_optimizers()
                <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
                method.
            version (str):
                Determines which ResNet version to use:

                - ``"std"``: original ResNet
                - ``"packed"``: Packed-Ensembles ResNet
                - ``"batched"``: BatchEnsemble ResNet
                - ``"masked"``: Masksemble ResNet
                - ``"mimo"``: MIMO ResNet
                - ``"mc-dropout"``: Monte-Carlo Dropout ResNet

            arch (int):
                Determines which ResNet architecture to use:

                - ``18``: ResNet-18
                - ``32``: ResNet-32
                - ``50``: ResNet-50
                - ``101``: ResNet-101
                - ``152``: ResNet-152

            style (str, optional): Which ResNet style to use. Defaults to
            ``imagenet``.
            normalization_layer (type[nn.Module], optional): Normalization layer
                to use. Defaults to ``nn.BatchNorm2d``.
            num_estimators (int, optional): Number of estimators in the ensemble.
                Only used if :attr:`version` is either ``"packed"``, ``"batched"``,
                ``"masked"`` or ``"mc-dropout"`` Defaults to ``None``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.
            mixup_params (dict, optional): Mixup parameters. Can include mixtype,
                mixmode, dist_sim, kernel_tau_max, kernel_tau_std,
                mixup_alpha, and cutmix_alpha. If None, no augmentations.
                Defaults to ``None``.
            width_multiplier (float, optional): Expansion factor affecting the width
                of the estimators. Defaults to ``1.0``
            groups (int, optional): Number of groups in convolutions. Defaults
                to ``1``.
            scale (float, optional): Expansion factor affecting the width of
                the estimators. Only used if :attr:`version` is ``"masked"``.
                Defaults to ``None``.
            last_layer_dropout (bool): whether to apply dropout to the last layer only.
            groups (int, optional): Number of groups in convolutions. Defaults to
                ``1``.
            scale (float, optional): Expansion factor affecting the width of the
                estimators. Only used if :attr:`version` is ``"masked"``. Defaults
                to ``None``.
            alpha (float, optional): Expansion factor affecting the width of the
                estimators. Only used if :attr:`version` is ``"packed"``. Defaults
                to ``None``.
            gamma (int, optional): Number of groups within each estimator. Only
                used if :attr:`version` is ``"packed"`` and scales with
                :attr:`groups`. Defaults to ``1``.
            rho (float, optional): Probability that all estimators share the same
                input. Only used if :attr:`version` is ``"mimo"``. Defaults to
                ``1``.
            batch_repeat (int, optional): Number of times to repeat the batch. Only
                used if :attr:`version` is ``"mimo"``. Defaults to ``1``.
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
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to
            ``False``.
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
            num_calibration_bins (int, optional): Number of calibration bins.
                Defaults to ``15``.
            pretrained (bool, optional): Indicates whether to use the pretrained
                weights or not. Only used if :attr:`version` is ``"packed"``.
                Defaults to ``False``.

        Raises:
            ValueError: If :attr:`version` is not either ``"std"``,
                ``"packed"``, ``"batched"``, ``"masked"`` or ``"mc-dropout"``.

        Returns:
            LightningModule: ResNet baseline ready for training and evaluation.
        """
        params = {
            "arch": arch,
            "conv_bias": False,
            "dropout_rate": dropout_rate,
            "groups": groups,
            "width_multiplier": width_multiplier,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
            "normalization_layer": normalization_layer,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        if version in ENSEMBLE_METHODS:
            params |= {
                "num_estimators": num_estimators,
            }

            if version != "mc-dropout":
                format_batch_fn = RepeatTarget(num_repeats=num_estimators)

        if version == "packed":
            params |= {
                "alpha": alpha,
                "gamma": gamma,
                "pretrained": pretrained,
            }

        elif version == "masked":
            params |= {
                "scale": scale,
            }

        elif version == "mimo":
            format_batch_fn = MIMOBatchFormat(
                num_estimators=num_estimators,
                rho=rho,
                batch_repeat=batch_repeat,
            )

        if version == "mc-dropout":  # std ResNets don't have `num_estimators`
            del params["num_estimators"]

        model = self.versions[version](**params)
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
            is_ensemble=version in ENSEMBLE_METHODS,
            optim_recipe=optim_recipe,
            format_batch_fn=format_batch_fn,
            mixup_params=mixup_params,
            eval_ood=eval_ood,
            eval_shift=eval_shift,
            eval_grouping_loss=eval_grouping_loss,
            ood_criterion=ood_criterion,
            log_plots=log_plots,
            save_in_csv=save_in_csv,
            calibration_set=calibration_set,
            num_calibration_bins=num_calibration_bins,
        )
        self.save_hyperparameters(ignore=["loss"])
