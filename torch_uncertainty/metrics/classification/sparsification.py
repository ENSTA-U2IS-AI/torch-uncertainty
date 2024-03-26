import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AUSE(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0
    plot_legend_name: str = "Sparsification Curves"

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        """The Area Under the Sparsification Error curve (AUSE) metric to estimate
        the quality of the uncertainty estimates, i.e., how much they coincide with
        the true errors.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            From the paper
            `Uncertainty estimates and multi-hypotheses for optical flow <https://arxiv.org/abs/1802.07095>`_.
            In ECCV, 2018.

        Inputs:
            - :attr:`scores`: Uncertainty scores of shape :math:`(B,)`. A higher
              score means a higher uncertainty.
            - :attr:`errors`: Binary errors of shape :math:`(B,)`,

        where :math:`B` is the batch size.

        Note:
            A higher AUSE means a lower quality of the uncertainty estimates.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, scores: Tensor, errors: Tensor) -> None:
        """Store the scores and their associated errors for later computation.

        Args:
            scores (Tensor): uncertainty scores of shape :math:`(B,)`
            errors (Tensor): binary errors of shape :math:`(B,)`
        """
        self.scores.append(scores)
        self.errors.append(errors)

    def compute(self) -> Tensor:
        """Compute the Area Under the Sparsification Error curve (AUSE) based
        on inputs passed to ``update``.

        Returns:
            Tensor: The AUSE.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        computed_error_rates = _rejection_rate_compute(scores, errors)
        computed_optimal_error_rates = _rejection_rate_compute(errors, errors)

        x = np.arange(computed_error_rates.size(0)) / computed_error_rates.size(
            0
        )
        y = (computed_error_rates - computed_optimal_error_rates).numpy()

        return torch.tensor([auc(x, y)])

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_oracle: bool = True,
        plot_value: bool = True,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the sparsification curve corresponding to the inputs passed to
        ``update``, and the oracle sparsification curve.

        Args:
            ax (Axes | None, optional): An matplotlib axis object. If provided
                will add plot to this axis. Defaults to None.
            plot_oracle (bool, optional): Whether to plot the oracle
                sparsification curve. Defaults to True.
            plot_value (bool, optional): Whether to plot the AUSE value.
                Defaults to True.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
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
        if plot_oracle:
            ax.plot(
                rejection_rates,
                computed_optimal_error_rates * 100,
                label="Oracle",
            )

        ax.set_xlabel("Rejection Rate (%)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlim(self.plot_lower_bound, self.plot_upper_bound)
        ax.set_ylim(self.plot_lower_bound, self.plot_upper_bound)
        ax.legend(loc="upper right")

        if plot_value:
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
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores (Tensor): uncertainty scores of shape :math:`(B,)`
        errors (Tensor): binary errors of shape :math:`(B,)`
    """
    num_samples = errors.size(0)

    order = scores.argsort()
    errors = errors[order]

    error_rates = torch.zeros(num_samples + 1)
    error_rates[0] = errors.sum()
    error_rates[1:] = errors.cumsum(dim=-1).flip(0)

    return error_rates / error_rates[0]
