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
    add_resnet_specific_args,
)
from torch_uncertainty.models.mc_dropout import mc_dropout
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
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.transforms import MIMOBatchFormat, RepeatTarget


class ResNet:
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

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        optimization_procedure: Any,
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
        num_estimators: int | None = None,
        dropout_rate: float = 0.0,
        last_layer_dropout: bool = False,
        groups: int = 1,
        scale: float | None = None,
        alpha: float | None = None,
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
                    "pretrained": pretrained,
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
        kwargs.update(params | {"version": version, "arch": arch})

        if version == "mc-dropout":  # std ResNets don't have `num_estimators`
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
        parser = add_resnet_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser = add_masked_specific_args(parser)
        parser = add_mimo_specific_args(parser)
        parser = add_mc_dropout_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="std",
            help=f"Variation of ResNet. Choose among: {cls.versions.keys()}",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action=BooleanOptionalAction,
            default=False,
        )
        return parser
