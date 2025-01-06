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
    r"""Routine for training & testing on **segmentation** tasks.

    Args:
        model (torch.nn.Module): Model to train.
        num_classes (int): Number of classes in the segmentation task.
        loss (torch.nn.Module): Loss function to optimize the :attr:`model`.
        optim_recipe (dict or Optimizer, optional): The optimizer and
            optionally the scheduler to use. Defaults to ``None``.
        eval_shift (bool, optional): Indicates whether to evaluate the Distribution
            shift performance. Defaults to ``False``.
        format_batch_fn (torch.nn.Module, optional): The function to format the
            batch. Defaults to ``None``.
        metric_subsampling_rate (float, optional): The rate of subsampling for the
            memory consuming metrics. Defaults to ``1e-2``.
        log_plots (bool, optional): Indicates whether to log plots from
            metrics. Defaults to ``False``.
        num_samples_to_plot (int, optional): Number of samples to plot in the
            segmentation results. Defaults to ``3``.
        num_calibration_bins (int, optional): Number of bins to compute calibration
            metrics. Defaults to ``15``.

    Warning:
        You must define :attr:`optim_recipe` if you do not use the CLI.

    Note:
        :attr:`optim_recipe` can be anything that can be returned by
        :meth:`LightningModule.configure_optimizers()`. Find more details
        `here <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers>`_.
    """

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
        ood_criterion: Literal["msp", "logit", "energy", "entropy", "mi", "vr"] = "msp",
        log_plots: bool = False,
        save_in_csv: bool = False,
        calibration_set: Literal["val", "test"] = "val",
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        num_calibration_bins: int = 15,
        pretrained: bool = False,
    ) -> None:
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
