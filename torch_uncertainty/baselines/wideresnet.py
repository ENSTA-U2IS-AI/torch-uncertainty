# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict

from torch_uncertainty.models.wideresnet.std import wideresnet28x10


# fmt:on
class WideResNetBaseline:
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        groups: int = 1,
        imagenet_structure: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        if self.groups < 1:
            raise ValueError("Number of groups must be at least 1.")

        self.groups = groups
        self.imagenet_structure = imagenet_structure

        self.model = wideresnet28x10(
            in_channels=in_channels,
            num_classes=num_classes,
            groups=groups,
            imagenet_structure=imagenet_structure,
        )

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        """Defines the model's attributes via command-line options:

        - ``--groups [int]``: defines :attr:`groups`. Defaults to ``1``.
        - ``--imagenet_structure``: sets :attr:`imagenet_structure`. Defaults
          to ``True``.
        """
        parent_parser.add_argument("--groups", type=int, default=1)
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        return parent_parser
