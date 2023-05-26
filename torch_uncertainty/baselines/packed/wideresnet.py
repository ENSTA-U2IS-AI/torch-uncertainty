# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict

import torch
import torch.nn as nn

from torch_uncertainty.models.wideresnet.packed import packed_wideresnet28x10
from torch_uncertainty.routines.classification import ClassificationEnsemble

# fmt: on


class PackedWideResNet(ClassificationEnsemble):
    r"""LightningModule for Packed-Ensembles WideResNet.

    Args:
        num_classes (int): Number of classes to predict.
        num_estimators (int): Number of estimators in the ensemble.
        in_channels (int): Number of input channels.
        alpha (int): Expansion factor affecting the width of the estimators.
        gamma (int): Number of groups within each estimator.
        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
    """

    def __init__(
        self,
        num_classes: int,
        num_estimators: int,
        in_channels: int,
        alpha: int,
        gamma: int,
        loss: nn.Module,
        optimization_procedure: Any,
        imagenet_structure: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            num_estimators=num_estimators,
            **kwargs,
        )

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

        self.loss = loss
        self.optimization_procedure = optimization_procedure

        self.model = packed_wideresnet28x10(
            in_channels=in_channels,
            num_estimators=num_estimators,
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            imagenet_structure=imagenet_structure,
        )

        # to log the graph
        self.example_input_array = torch.randn(1, in_channels, 32, 32)

    def configure_optimizers(self) -> dict:
        """Configures the optimizers.

        Returns:
            dict: Optimizers.
        """
        return self.optimization_procedure(self)

    @property
    def criterion(self) -> nn.Module:
        return self.loss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model.forward(input)

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--num_estimators [int]``: defines :attr:`num_estimators`. Defaults
          to ``1``.
        - ``--imagenet_structure``: sets :attr:`imagenet_structure`. Defaults
          to ``True``.

        Example:

            .. parsed-literal::

                python script.py --num_estimators 4 --alpha 2
        """
        parent_parser = ClassificationEnsemble.add_model_specific_args(
            parent_parser
        )
        parent_parser.add_argument("--num_estimators", type=int, default=4)
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        parent_parser.add_argument("--alpha", type=int, default=2)
        parent_parser.add_argument("--gamma", type=int, default=1)
        return parent_parser
