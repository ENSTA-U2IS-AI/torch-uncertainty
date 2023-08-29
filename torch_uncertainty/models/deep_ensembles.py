# fmt: off
import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn


# fmt: on
class _DeepEnsembles(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
    ) -> None:
        super().__init__()

        self.models = nn.ModuleList(models)
        self.num_estimators = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits of the ensemble

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor: The output of the model with shape :math:`(B, N, C)`,
                where :math:`B` is the batch size, :math:`N` is the number of
                estimators, and :math:`C` is the number of classes.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.forward(x))
        return torch.stack(predictions, dim=1)


def deep_ensembles(
    models: Union[List[nn.Module], nn.Module],
    num_estimators: Optional[int] = None,
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
            is a module (or singleton list).
        ValueError: If :attr:num_estimators is less than 2 and :attr:models is
            a module (or singleton list).
        ValueError: If :attr:num_estimators is defined while :attr:models is
            a (non-singleton) list.

    References:
        Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
        Simple and scalable predictive uncertainty estimation using deep
        ensembles. In NeurIPS, 2017.
    """
    if (
        isinstance(models, list)
        and len(models) == 1
        or isinstance(models, nn.Module)
    ):
        if num_estimators is None:
            raise ValueError(
                "if models is a module, num_estimators must be specified."
            )
        elif num_estimators < 2:
            raise ValueError(
                "num_estimators must be at least 2. Got" f"{num_estimators}."
            )

        if isinstance(models, list):
            models = models[0]

        models = [copy.deepcopy(models) for _ in range(num_estimators)]
    elif (
        isinstance(models, list)
        and len(models) > 1
        and num_estimators is not None
    ):
        raise ValueError(
            "num_estimators must be None if you provided a non-singleton list."
        )

    return _DeepEnsembles(models=models)
