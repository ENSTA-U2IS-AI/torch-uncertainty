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

from ...models.wideresnet import (
    batched_wideresnet28x10,
    masked_wideresnet28x10,
    mimo_wideresnet28x10,
    packed_wideresnet28x10,
    wideresnet28x10,
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
    add_wideresnet_specific_args,
)


# fmt: on
class WideResNet:
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

            - ``"vanilla"``: original Wide-ResNet
            - ``"packed"``: Packed-Ensembles Wide-ResNet
            - ``"batched"``: BatchEnsemble Wide-ResNet
            - ``"masked"``: Masksemble Wide-ResNet
            - ``"mimo"``: MIMO ResNet

        style (bool, optional): (str, optional): Which ResNet style to use.
        Defaults to ``imagenet``.
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

    Raises:
        ValueError: If :attr:`version` is not either ``"vanilla"``,
            ``"packed"``, ``"batched"`` or ``"masked"``.

    Returns:
        LightningModule: Wide-ResNet baseline ready for training and
            evaluation.
    """
    single = ["vanilla"]
    ensemble = ["packed", "batched", "masked", "mimo"]
    versions = {
        "vanilla": [wideresnet28x10],
        "packed": [packed_wideresnet28x10],
        "batched": [batched_wideresnet28x10],
        "masked": [masked_wideresnet28x10],
        "mimo": [mimo_wideresnet28x10],
    }

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: Type[nn.Module],
        optimization_procedure: Any,
        version: Literal["vanilla", "packed", "batched", "masked", "mimo"],
        style: str = "imagenet",
        num_estimators: Optional[int] = None,
        groups: Optional[int] = None,
        scale: Optional[float] = None,
        alpha: Optional[int] = None,
        gamma: Optional[int] = None,
        rho: float = 1.0,
        batch_repeat: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        # pretrained: bool = False,
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

        # version specific params
        if version == "packed":
            params.update(
                {
                    "num_estimators": num_estimators,
                    "alpha": alpha,
                    "gamma": gamma,
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

        model = cls.versions[version][0](**params)
        kwargs.update(params)
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
        parser = add_wideresnet_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser = add_masked_specific_args(parser)
        parser = add_mimo_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help="Variation of WideResNet. "
            + f"Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )

        return parser
