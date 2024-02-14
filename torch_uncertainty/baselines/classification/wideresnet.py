from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv,
    load_hparams_from_yaml,
)
from torch import nn

from torch_uncertainty.baselines.utils.parser_addons import (
    add_masked_specific_args,
    add_mc_dropout_specific_args,
    add_mimo_specific_args,
    add_packed_specific_args,
    add_wideresnet_specific_args,
)
from torch_uncertainty.models.mc_dropout import mc_dropout
from torch_uncertainty.models.wideresnet import (
    batched_wideresnet28x10,
    masked_wideresnet28x10,
    mimo_wideresnet28x10,
    packed_wideresnet28x10,
    wideresnet28x10,
)
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.transforms import MIMOBatchFormat, RepeatTarget


class WideResNet:
    single = ["std"]
    ensemble = ["packed", "batched", "masked", "mc-dropout", "mimo"]
    versions = {
        "std": [wideresnet28x10],
        "mc-dropout": [wideresnet28x10],
        "packed": [packed_wideresnet28x10],
        "batched": [batched_wideresnet28x10],
        "masked": [masked_wideresnet28x10],
        "mimo": [mimo_wideresnet28x10],
    }

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        optimization_procedure: Any,
        version: Literal[
            "std", "mc-dropout", "packed", "batched", "masked", "mimo"
        ],
        style: str = "imagenet",
        num_estimators: int | None = None,
        dropout_rate: float = 0.0,
        last_layer_dropout: bool = False,
        groups: int | None = None,
        scale: float | None = None,
        alpha: int | None = None,
        gamma: int | None = None,
        rho: float = 1.0,
        batch_repeat: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        # pretrained: bool = False,
        **kwargs,
    ) -> LightningModule:
        r"""Wide-ResNet28x10 backbone baseline for classification providing support
        for various versions.

        Args:
            num_classes (int): Number of classes to predict.
            in_channels (int): Number of input channels.
            loss (nn.Module): Training loss.
            optimization_procedure (Any): Optimization procedure, corresponds to
                what expect the `LightningModule.configure_optimizers()
                <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
                method.
            version (str):
                Determines which Wide-ResNet version to use:

                - ``"std"``: original Wide-ResNet
                - ``"mc-dropout"``: Monte Carlo Dropout Wide-ResNet
                - ``"packed"``: Packed-Ensembles Wide-ResNet
                - ``"batched"``: BatchEnsemble Wide-ResNet
                - ``"masked"``: Masksemble Wide-ResNet
                - ``"mimo"``: MIMO Wide-ResNet

            style (bool, optional): (str, optional): Which ResNet style to use.
            Defaults to ``imagenet``.
            num_estimators (int, optional): Number of estimators in the ensemble.
                Only used if :attr:`version` is either ``"packed"``, ``"batched"``
                or ``"masked"`` Defaults to ``None``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.
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
                :attr:`groups`. Defaults to ``1s``.
            rho (float, optional): Probability that all estimators share the same
                input. Only used if :attr:`version` is ``"mimo"``. Defaults to
                ``1``.
            batch_repeat (int, optional): Number of times to repeat the batch. Only
                used if :attr:`version` is ``"mimo"``. Defaults to ``1``.
            use_entropy (bool, optional): Indicates whether to use the entropy
                values as the OOD criterion or not. Defaults to ``False``.
            use_logits (bool, optional): Indicates whether to use the logits as the
                OOD criterion or not. Defaults to ``False``.
            use_mi (bool, optional): Indicates whether to use the mutual
                information as the OOD criterion or not. Defaults to ``False``.
            use_variation_ratio (bool, optional): Indicates whether to use the
                variation ratio as the OOD criterion or not. Defaults to ``False``.
            pretrained (bool, optional): Indicates whether to use the pretrained
                weights or not. Only used if :attr:`version` is ``"packed"``.
                Defaults to ``False``.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If :attr:`version` is not either ``"std"``,
                ``"packed"``, ``"batched"`` or ``"masked"``.

        Returns:
            LightningModule: Wide-ResNet baseline ready for training and
                evaluation.
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

        if version not in cls.versions:
            raise ValueError(f"Unknown version: {version}")

        if version in cls.ensemble:
            params.update(
                {
                    "num_estimators": num_estimators,
                }
            )
            if version != "mc-dropout":
                format_batch_fn = RepeatTarget(num_repeats=num_estimators)

        if version == "packed":
            params.update(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                }
            )
        elif version == "masked":
            params.update(
                {
                    "scale": scale,
                }
            )
        elif version == "mimo":
            format_batch_fn = MIMOBatchFormat(
                num_estimators=num_estimators,
                rho=rho,
                batch_repeat=batch_repeat,
            )

        # for lightning params
        kwargs.update(params | {"version": version})

        if version == "mc-dropout":  # std wideRn don't have `num_estimators`
            del params["num_estimators"]
        model = cls.versions[version][0](**params)
        if version == "mc-dropout":
            model = mc_dropout(
                model=model,
                num_estimators=num_estimators,
                last_layer=last_layer_dropout,
            )

        # routine specific parameters
        if version in cls.single:
            return ClassificationSingle(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                format_batch_fn=format_batch_fn,
                use_entropy=use_entropy,
                use_logits=use_logits,
                **kwargs,
            )
        # version in cls.ensemble
        return ClassificationEnsemble(
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            format_batch_fn=format_batch_fn,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
            **kwargs,
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        hparams_file: str | Path,
        **kwargs,
    ) -> LightningModule:  # coverage: ignore
        if hparams_file is not None:
            extension = str(hparams_file).split(".")[-1]
            if extension.lower() == "csv":
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ("yml", "yaml"):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError(
                    ".csv, .yml or .yaml is required for `hparams_file`"
                )

        hparams.update(kwargs)
        checkpoint = torch.load(checkpoint_path)
        obj = cls(**hparams)
        obj.load_state_dict(checkpoint["state_dict"])
        return obj

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = ClassificationEnsemble.add_model_specific_args(parser)
        parser = add_wideresnet_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser = add_masked_specific_args(parser)
        parser = add_mimo_specific_args(parser)
        parser = add_mc_dropout_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="std",
            help=f"Variation of WideResNet. Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )

        return parser
