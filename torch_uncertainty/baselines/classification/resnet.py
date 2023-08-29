# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv,
    load_hparams_from_yaml,
)

from ...models.resnet import (
    batched_resnet18,
    batched_resnet34,
    batched_resnet50,
    batched_resnet101,
    batched_resnet152,
    masked_resnet18,
    masked_resnet34,
    masked_resnet50,
    masked_resnet101,
    masked_resnet152,
    mimo_resnet18,
    mimo_resnet34,
    mimo_resnet50,
    mimo_resnet101,
    mimo_resnet152,
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from ...routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from ...transforms import MIMOBatchFormat, RepeatTarget
from ..utils.parser_addons import (
    add_masked_specific_args,
    add_mimo_specific_args,
    add_packed_specific_args,
    add_resnet_specific_args,
)


# fmt: on
class ResNet:
    r"""ResNet backbone baseline for classification providing support for
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
            Determines which ResNet version to use:

            - ``"vanilla"``: original ResNet
            - ``"packed"``: Packed-Ensembles ResNet
            - ``"batched"``: BatchEnsemble ResNet
            - ``"masked"``: Masksemble ResNet
            - ``"mimo"``: MIMO ResNet

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
            Only used if :attr:`version` is either ``"packed"``, ``"batched"``
            or ``"masked"`` Defaults to ``None``.
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

    Raises:
        ValueError: If :attr:`version` is not either ``"vanilla"``,
            ``"packed"``, ``"batched"`` or ``"masked"``.

    Returns:
        LightningModule: ResNet baseline ready for training and evaluation.
    """

    single = ["vanilla"]
    ensemble = ["packed", "batched", "masked", "mimo"]
    versions = {
        "vanilla": [resnet18, resnet34, resnet50, resnet101, resnet152],
        "packed": [
            packed_resnet18,
            packed_resnet34,
            packed_resnet50,
            packed_resnet101,
            packed_resnet152,
        ],
        "batched": [
            batched_resnet18,
            batched_resnet34,
            batched_resnet50,
            batched_resnet101,
            batched_resnet152,
        ],
        "masked": [
            masked_resnet18,
            masked_resnet34,
            masked_resnet50,
            masked_resnet101,
            masked_resnet152,
        ],
        "mimo": [
            mimo_resnet18,
            mimo_resnet34,
            mimo_resnet50,
            mimo_resnet101,
            mimo_resnet152,
        ],
    }
    archs = [18, 34, 50, 101, 152]

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: Type[nn.Module],
        optimization_procedure: Any,
        version: Literal["vanilla", "packed", "batched", "masked", "mimo"],
        arch: int,
        style: str = "imagenet",
        num_estimators: Optional[int] = None,
        dropout_rate: float = 0.0,
        groups: int = 1,
        scale: Optional[float] = None,
        alpha: Optional[float] = None,
        gamma: int = 1,
        rho: float = 1.0,
        batch_repeat: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        pretrained: bool = False,
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "style": style,
            "groups": groups,
        }

        format_batch_fn = nn.Identity()

        if version not in cls.versions.keys():
            raise ValueError(f"Unknown version: {version}")

        if version == "vanilla":
            params.update(
                {
                    "dropout_rate": dropout_rate,
                }
            )
        elif version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "gamma": gamma,
                    "pretrained": pretrained,
                }
            )
            format_batch_fn = RepeatTarget(num_repeats=num_estimators)
        elif version == "batched":
            params.update(
                {
                    "num_estimators": num_estimators,
                }
            )
            format_batch_fn = RepeatTarget(num_repeats=num_estimators)
        elif version == "masked":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "scale": scale,
                }
            )
            format_batch_fn = RepeatTarget(num_repeats=num_estimators)
        elif version == "mimo":
            params.update(
                {
                    "num_estimators": num_estimators,
                }
            )
            format_batch_fn = MIMOBatchFormat(
                num_estimators=num_estimators,
                rho=rho,
                batch_repeat=batch_repeat,
            )

        model = cls.versions[version][cls.archs.index(arch)](**params)
        kwargs.update(params)
        kwargs.update({"version": version, "arch": arch})
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
        elif version in cls.ensemble:
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
        parser = add_resnet_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser = add_masked_specific_args(parser)
        parser = add_mimo_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help=f"Variation of ResNet. Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )
        return parser
