from typing import Literal

from torch import nn

from torch_uncertainty.models import mc_dropout
from torch_uncertainty.models.resnet import (
    batched_resnet18,
    batched_resnet20,
    batched_resnet34,
    batched_resnet50,
    batched_resnet101,
    batched_resnet152,
    masked_resnet18,
    masked_resnet20,
    masked_resnet34,
    masked_resnet50,
    masked_resnet101,
    masked_resnet152,
    mimo_resnet18,
    mimo_resnet20,
    mimo_resnet34,
    mimo_resnet50,
    mimo_resnet101,
    mimo_resnet152,
    packed_resnet18,
    packed_resnet20,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
    resnet18,
    resnet20,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.transforms import MIMOBatchFormat, RepeatTarget


class ResNetBaseline(ClassificationRoutine):
    single = ["std"]
    ensemble = ["packed", "batched", "masked", "mc-dropout", "mimo"]
    versions = {
        "std": [
            resnet18,
            resnet20,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
        ],
        "packed": [
            packed_resnet18,
            packed_resnet20,
            packed_resnet34,
            packed_resnet50,
            packed_resnet101,
            packed_resnet152,
        ],
        "batched": [
            batched_resnet18,
            batched_resnet20,
            batched_resnet34,
            batched_resnet50,
            batched_resnet101,
            batched_resnet152,
        ],
        "masked": [
            masked_resnet18,
            masked_resnet20,
            masked_resnet34,
            masked_resnet50,
            masked_resnet101,
            masked_resnet152,
        ],
        "mimo": [
            mimo_resnet18,
            mimo_resnet20,
            mimo_resnet34,
            mimo_resnet50,
            mimo_resnet101,
            mimo_resnet152,
        ],
        "mc-dropout": [
            resnet18,
            resnet20,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
        ],
    }
    archs = [18, 20, 34, 50, 101, 152]

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
            "masked",
            "mimo",
        ],
        arch: int,
        style: str = "imagenet",
        num_estimators: int = 1,
        dropout_rate: float = 0.0,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1.0,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
        last_layer_dropout: bool = False,
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
        calibration_set: Literal["val", "test"] | None = None,
        eval_ood: bool = False,
        eval_grouping_loss: bool = False,
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
            num_estimators (int, optional): Number of estimators in the ensemble.
                Only used if :attr:`version` is either ``"packed"``, ``"batched"``,
                ``"masked"`` or ``"mc-dropout"`` Defaults to ``None``.
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
            eval_grouping_loss (bool, optional): Indicates whether to evaluate the
                grouping loss or not. Defaults to ``False``.
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
            "conv_bias": False,
            "dropout_rate": dropout_rate,
            "groups": groups,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")

        if version in self.ensemble:
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
            eval_grouping_loss=eval_grouping_loss,
            ood_criterion=ood_criterion,
            log_plots=log_plots,
            save_in_csv=save_in_csv,
            calibration_set=calibration_set,
        )
        self.save_hyperparameters(ignore=["loss"])
