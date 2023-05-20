# fmt: off
from argparse import ArgumentParser
from typing import Any, Dict


# fmt:on
class PackedBaseline:
    def __init__(
        self,
        num_estimators: int,
        alpha: int,
        gamma: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"Attribute `alpha` should be > 0, not {alpha}")
        if gamma < 1:
            raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

        self.alpha = alpha
        self.gamma = gamma
        self.num_estimators = num_estimators

        # construct config
        self.save_hyperparameters(ignore=["loss", "optimization_procedure"])

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        parent_parser.add_argument("--num_estimators", type=int, default=4)
        parent_parser.add_argument("--alpha", type=int, default=2)
        parent_parser.add_argument("--gamma", type=int, default=1)
        return parent_parser
