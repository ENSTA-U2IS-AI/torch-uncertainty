from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from sklearn.metrics import auc


class AUSE(Metric):
    """The Area Under the Sparsification Error curve (AUSE) metric to estimate
    the quality of the uncertainty estimates, i.e., how much they coincide with
    the true errors.

    Args:
        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - :attr:`scores`: Uncertainty scores of shape :math:`(B,)`. A higher
            score means a higher uncertainty.
        - :attr:`errors`: Errors of shape :math:`(B,)`.

        where :math:`B` is the batch size.

    Note:
        A higher AUSE means a lower quality of the uncertainty estimates.

    References:
        Eddy Ilg, Özgün Çiçek, Silvio Galesso, Aaron Klein, Osama Makansi,
        Frank Hutter, and Thomas Brox. Uncertainty estimates and
        multi-hypotheses for optical flow. In ECCV, 2018.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0
    plot_legend_name: str = "Sparsification Curves"

    scores: List[Tensor]
    errors: List[Tensor]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, scores: Tensor, errors: Tensor) -> None:
        self.scores.append(scores)
        self.errors.append(errors)

    def compute(self) -> Tensor:
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        computed_error_rates = _rejection_rate_compute(scores, errors)
        computed_optimal_error_rates = _rejection_rate_compute(errors, errors)

        x = np.arange(computed_error_rates.size(0)) / computed_error_rates.size(
            0
        )
        y = (computed_error_rates - computed_optimal_error_rates).numpy()

        return torch.tensor([auc(x, y)])

    def plot(self, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        fig, ax = plt.subplots() if ax is None else (None, ax)

        # Computation of AUSEC
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        computed_error_rates = _rejection_rate_compute(scores, errors)
        computed_optimal_error_rates = _rejection_rate_compute(errors, errors)

        x = np.arange(computed_error_rates.size(0)) / computed_error_rates.size(
            0
        )
        y = (computed_error_rates - computed_optimal_error_rates).numpy()

        ausec = auc(x, y)

        rejection_rates = (
            np.arange(computed_error_rates.size(0))
            / computed_error_rates.size(0)
        ) * 100

        ax.plot(
            rejection_rates,
            computed_error_rates * 100,
            label="Model",
        )
        ax.plot(
            rejection_rates,
            computed_optimal_error_rates * 100,
            label="Optimal",
        )

        ax.set_xlabel("Rejection Rate (%)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlim(self.plot_lower_bound, self.plot_upper_bound)
        ax.set_ylim(self.plot_lower_bound, self.plot_upper_bound)
        ax.legend(loc="upper right")

        ax.text(
            0.02,
            0.02,
            f"AUSEC={ausec:.03}",
            color="black",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        return fig, ax


def _rejection_rate_compute(
    errors: Tensor,
    uncertainty_score: Tensor,
) -> Tensor:
    num_samples = errors.size(0)

    order = uncertainty_score.argsort()
    errors = errors[order]

    error_rates = torch.zeros(num_samples + 1)
    error_rates[0] = errors.sum()
    error_rates[1:] = errors.cumsum(dim=-1).flip(0)

    return error_rates / error_rates[0]
