# fmt: off
import copy
from typing import List, Union

import torch
import torch.nn as nn


# fmt: on
class DeepEnsembles(nn.Module):
    def __init__(
        self,
        in_channels: int,
        models: List[nn.Module],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.models = models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for model in self.models:
            predictions.append(model.forward(x))
        return torch.stack(predictions, dim=0)


def deep_ensembles(
    models: Union[List[nn.Module], nn.Module],
    num_estimators: int = None,
) -> nn.Module:
    """
    Builds a Deep Ensembles out of the original models.

    Args:
        model (nn.Module): The model to be ensembled.
        num_estimators (int): The number of estimators in the ensemble.

    Returns:
        nn.Module: The ensembled model.

    Raises:
        ValueError: If :attr:num_estimators is not specified and :attr:models
            is a module.
        ValueError: If :attr:num_estimators is less than 2 and :attr:models is
            a module.
    """
    if isinstance(models, list) and len(models) == 1:
        if num_estimators is None:
            raise ValueError(
                "if models is a module, num_estimators must be specified."
            )
        models = models[0]

    if isinstance(models, nn.Module) and num_estimators is not None:
        if num_estimators < 2:
            raise ValueError("num_estimators must be at least 2.")
        models = [copy.deepcopy(models) for _ in range(num_estimators)]

    return DeepEnsembles(models=models)