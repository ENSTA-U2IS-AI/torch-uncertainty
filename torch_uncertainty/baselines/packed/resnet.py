# fmt: off
from argparse import ArgumentParser
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

from torch_uncertainty.models.resnet import (
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
)
from torch_uncertainty.routines.classification import ClassificationEnsemble

# fmt: on
archs = [
    packed_resnet18,
    packed_resnet34,
    packed_resnet50,
    packed_resnet101,
    packed_resnet152,
]
choices = [18, 34, 50, 101, 152]


class PackedResNet(ClassificationEnsemble):
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

    Note:

        The OOD criterion is by defaults the confidence score.

    Warning:

        Make sure at most only one of :attr:`use_entropy`, :attr:`use_logits`,
        :attr:`use_mi` and :attr:`use_variation_ratio` attributes is set to
        ``True``. Otherwise a :class:`ValueError()` will be raised.

    Raises:
            ValueError: If :attr:`alpha`:math:`<1`.
            ValueError: If :attr:`gamma`:math:`<1`.
    """

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
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_estimators=num_estimators,
            use_entropy=use_entropy,
            use_logits=use_logits,
            use_mi=use_mi,
            use_variation_ratio=use_variation_ratio,
        )

        if alpha <= 0:
            raise ValueError(f"Attribute `alpha` should be > 0, not {alpha}")
        if gamma < 1:
            raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = archs[choices.index(arch)](
            in_channels=in_channels,
            num_estimators=num_estimators,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    def configure_optimizers(self) -> dict:
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        input = input.repeat(1, self.num_estimators, 1, 1)
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--arch [int]``: defines :attr:`arch`. Defaults to ``18``.
        - ``--num_estimators [int]``: defines :attr:`num_estimators`. Defaults
          to ``1``.
        - ``--alpha [int]``: defines :attr:`alpha`. Defaults to ``1``.
        - ``--gamma [int]``: defines :attr:`gamma`. Defaults to ``1``.
        - ``--entropy``: sets :attr:`use_entropy` to ``True``.
        - ``--logits``: sets :attr:`use_logits` to ``True``.
        - ``--mutual_information``: sets :attr:`use_mi` to ``True``.
        - ``--variation_ratio``: sets :attr:`use_variation_ratio` to ``True``.

        Example:

            .. parsed-literal::

                python script.py --arch 18 --num_estimators 4 --alpha 2
        """
        parent_parser.add_argument(
            "--arch",
            type=int,
            default=18,
            choices=choices,
            help="Type of ResNet",
        )
        parent_parser.add_argument("--alpha", type=int, default=2)
        parent_parser.add_argument(
            "--entropy", dest="use_entropy", action="store_true"
        )
        parent_parser.add_argument("--gamma", type=int, default=1)
        parent_parser.add_argument(
            "--logits", dest="use_logits", action="store_true"
        )
        parent_parser.add_argument(
            "--mutual_information", dest="uses_mi", action="store_true"
        )
        parent_parser.add_argument("--num_estimators", type=int, default=4)
        parent_parser.add_argument(
            "--variation_ratio", dest="use_variation_ratio", action="store_true"
        )
        return parent_parser
