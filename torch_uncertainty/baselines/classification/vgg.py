from argparse import ArgumentParser
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
    add_mc_dropout_specific_args,
    add_packed_specific_args,
    add_vgg_specific_args,
)
from torch_uncertainty.models.mc_dropout import mc_dropout
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
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.transforms import RepeatTarget


class VGG:
    single = ["std"]
    ensemble = ["packed", "mc-dropout"]
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

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        optimization_procedure: Any,
        version: Literal["std", "mc-dropout", "packed"],
        arch: int,
        num_estimators: int | None = None,
        dropout_rate: float = 0.0,
        last_layer_dropout: bool = False,
        style: str = "imagenet",
        groups: int = 1,
        alpha: float | None = None,
        gamma: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        **kwargs,
    ) -> LightningModule:
        r"""VGG backbone baseline for classification providing support for
        various versions and architectures.

        Args:
            num_classes (int): Number of classes to predict.
            in_channels (int): Number of input channels.
            loss (nn.Module): Training loss.
            optimization_procedure (Any): Optimization procedure, corresponds to
                what expect the `LightningModule.configure_optimizers()
                <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
                method.
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
            last_layer_dropout (bool): whether to apply dropout to the last layer only.
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
            **kwargs: Additional arguments to be passed to the
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

        if version not in cls.versions:
            raise ValueError(f"Unknown version: {version}")

        format_batch_fn = nn.Identity()

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
                    "style": style,
                    "gamma": gamma,
                }
            )

        # for lightning params
        kwargs.update(params | {"version": version, "arch": arch})

        if version == "mc-dropout":  # std VGGs don't have `num_estimators`
            del params["num_estimators"]
        model = cls.versions[version][cls.archs.index(arch)](**params)
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
        parser = add_vgg_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser = add_mc_dropout_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="std",
            help=f"Variation of VGG. Choose among: {cls.versions.keys()}",
        )
        return parser
