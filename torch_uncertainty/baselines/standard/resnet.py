# fmt: off
from argparse import ArgumentParser
from typing import Any

import torch
import torch.nn as nn

from torch_uncertainty.baselines.resnet import ResNetBaseline
from torch_uncertainty.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch_uncertainty.routines.classification import ClassificationSingle

# fmt: on
archs = [resnet18, resnet34, resnet50, resnet101, resnet152]
choices = [18, 34, 50, 101, 152]


class ResNet(ClassificationSingle, ResNetBaseline):
    r"""LightningModule for Vanilla ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        in_channels (int): Number of input channels.
        arch (int):
            Determines which ResNet architecture to use:

            - ``18``: ResNet-18
            - ``32``: ResNet-32
            - ``50``: ResNet-50
            - ``101``: ResNet-101
            - ``152``: ResNet-152

        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
        groups (int, optional): Number of groups in convolutions. Defaults to
            ``1``.
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.
        imagenet_structure (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.

    Note:
        The default OOD criterion is the maximum softmax score.

    Warning:
        Make sure at most only one of :attr:`use_entropy` and :attr:`use_logits`
        attributes is set to ``True``. Otherwise a :class:`ValueError()` will
        be raised.

    Raises:
        ValueError: If :attr:`groups` :math:`<1`.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        arch: int,
        loss: nn.Module,
        optimization_procedure: Any,
        groups: int = 1,
        use_entropy: bool = False,
        use_logits: bool = False,
        imagenet_structure: bool = True,
        **kwargs,
    ) -> None:
        model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_classes=num_classes,
            groups=groups,
            imagenet_structure=imagenet_structure,
        )
        ClassificationSingle.__init__(
            self=self,
            num_classes=num_classes,
            in_channels=in_channels,
            arch=arch,
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            groups=groups,
            use_entropy=use_entropy,
            use_logits=use_logits,
        )
        ResNetBaseline.__init__(
            self=self,
            num_classes=num_classes,
            in_channels=in_channels,
            arch=arch,
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            groups=groups,
            use_entropy=use_entropy,
            use_logits=use_logits,
        )

        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parent_parser = ClassificationSingle.add_model_specific_args(
            parent_parser
        )
        return ResNetBaseline.add_model_specific_args(parent_parser)
