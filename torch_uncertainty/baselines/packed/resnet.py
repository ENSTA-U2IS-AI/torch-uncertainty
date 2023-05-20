# fmt: off
from argparse import ArgumentParser
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

from torch_uncertainty.baselines.packed.packed import PackedBaseline
from torch_uncertainty.baselines.resnet import ResNetBaseline
from torch_uncertainty.models.resnet import (
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
)
from torch_uncertainty.routines.classification import ClassificationEnsemble
from torch_uncertainty.utils import load_hf

# fmt: on
archs = [
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
]
choices = [18, 34, 50, 101, 152]

weight_ids = {
    "10": {
        "18": None,
        "32": None,
        "50": "pe_resnet50_c10",
        "101": None,
        "152": None,
    },
    "100": {
        "18": None,
        "32": None,
        "50": "pe_resnet50_c100",
        "101": None,
        "152": None,
    },
    "1000": {
        "18": None,
        "32": None,
        "50": "pe_resnet50_in1k",
        "101": None,
        "152": None,
    },
    "1000_wider": {
        "18": None,
        "32": None,
        "50": "pex4_resnet50",
        "101": None,
        "152": None,
    },
}


class PackedResNet(ClassificationEnsemble, ResNetBaseline, PackedBaseline):
    r"""LightningModule for Packed-Ensembles ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        in_channels (int): Number of input channels.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
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
        use_entropy (bool, optional): Indicates whether to use the entropy
            values as the OOD criterion or not. Defaults to ``False``.
        use_logits (bool, optional): Indicates whether to use the logits as the
            OOD criterion or not. Defaults to ``False``.
        use_mi (bool, optional): Indicates whether to use the mutual
            information as the OOD criterion or not. Defaults to ``False``.
        use_variation_ratio (bool, optional): Indicates whether to use the
            variation ratio as the OOD criterion or not. Defaults to ``False``.
        pretrained (bool, optional): Indicates whether to use the pretrained
            weights or not. Defaults to ``False``.

    Note:
        The default OOD criterion is the maximum softmax score.

    Warning:
        Make sure at most only one of :attr:`use_entropy`, :attr:`use_logits`,
        :attr:`use_mi` and :attr:`use_variation_ratio` attributes is set to
        ``True``. Otherwise a :class:`ValueError()` will be raised.

    Raises:
        ValueError: If :attr:`alpha`:math:`\leq 0`.
        ValueError: If :attr:`gamma`:math:`<1`.
    """

    weights_id = "torch-uncertainty/pe_resnet50_in1k"

    def __init__(
        self,
        num_classes: int,
        num_estimators: int,
        in_channels: int,
        alpha: int,
        gamma: int,
        arch: Literal[18, 34, 50, 101, 152],
        loss: nn.Module,
        optimization_procedure: Any,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        imagenet_structure: bool = True,
        pretrained: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_estimators=num_estimators,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            imagenet_structure=imagenet_structure,
        )
        ClassificationEnsemble.__init__(
            self=self,
            model=model,
            num_classes=num_classes,
            num_estimators=num_estimators,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
        )
        ResNetBaseline.__init__(
            self=self,
            num_classes=num_classes,
            num_estimators=num_estimators,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
        )
        PackedBaseline.__init__(
            self=self, alpha=alpha, gamma=gamma, num_estimators=num_estimators
        )

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

        self._load(pretrained, arch, num_classes)

    def _load(self, pretrained: bool, arch: str, num_classes: int):
        if pretrained:
            weights = weight_ids[str(num_classes)][arch]
            if weights is None:
                raise ValueError("No pretrained weights for this configuration")
            self.model.load_state_dict(load_hf(weights))

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parent_parser = PackedBaseline.add_model_specific_args(parent_parser)
        parent_parser = ResNetBaseline.add_model_specific_args(parent_parser)
        parent_parser = ClassificationEnsemble.add_model_specific_args(
            parent_parser
        )
        return parent_parser
