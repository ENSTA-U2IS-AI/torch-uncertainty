# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv,
    load_hparams_from_yaml,
)

from ...models.vgg import (
    packed_vgg11,
    packed_vgg13,
    packed_vgg16,
    packed_vgg19,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
)
from ...routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from ..utils.parser_addons import (
    add_packed_specific_args,
    add_vgg_specific_args,
)


# fmt: on
class VGG:
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

            - ``"vanilla"``: original VGG
            - ``"packed"``: Packed-Ensembles VGG
            - ``"batched"``: BatchEnsemble VGG

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

    Raises:
        ValueError: If :attr:`version` is not either ``"vanilla"``,
            ``"packed"``, ``"batched"`` or ``"masked"``.

    Returns:
        LightningModule: VGG baseline ready for training and evaluation.
    """

    single = ["vanilla"]
    ensemble = ["packed", "batched"]
    versions = {
        "vanilla": [vgg11, vgg13, vgg16, vgg19],
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
        loss: Type[nn.Module],
        optimization_procedure: Any,
        version: Literal["vanilla", "packed"],
        arch: int,
        num_estimators: Optional[int] = None,
        style: str = "imagenet",
        groups: int = 1,
        alpha: Optional[float] = None,
        gamma: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
            "groups": groups,
        }

        if version not in cls.versions.keys():
            raise ValueError(f"Unknown version: {version}")

        if version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "style": style,
                    "gamma": gamma,
                }
            )

        model = cls.versions[version][cls.archs.index(arch)](**params)
        kwargs.update(params)
        # routine specific parameters
        if version in cls.single:
            return ClassificationSingle(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                use_entropy=use_entropy,
                use_logits=use_logits,
                **kwargs,
            )
        elif version in cls.ensemble:
            return ClassificationEnsemble(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                use_entropy=use_entropy,
                use_logits=use_logits,
                use_mi=use_mi,
                use_variation_ratio=use_variation_ratio,
                **kwargs,
            )
        else:
            raise ValueError(
                f"{version} is not in {cls.single} nor {cls.ensemble}."
            )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        hparams_file: Union[str, Path],
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
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help=f"Variation of VGG. Choose among: {cls.versions.keys()}",
        )
        return parser
