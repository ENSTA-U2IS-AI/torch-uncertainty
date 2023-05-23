# fmt: off
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict

# fmt:on
choices = [18, 34, 50, 101, 152]


class ResNetBaseline:
    def __init__(
        self,
        groups: int = 1,
        imagenet_structure: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        if groups < 1:
            raise ValueError("Number of groups must be at least 1.")

        self.groups = groups
        self.imagenet_structure = imagenet_structure

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
            "--arch",
            type=int,
            default=18,
            choices=choices,
            help="Type of ResNet",
        )
        parent_parser.add_argument(
            "--imagenet_structure",
            action=BooleanOptionalAction,
            default=True,
            help="Use imagenet structure",
        )
        return parent_parser
