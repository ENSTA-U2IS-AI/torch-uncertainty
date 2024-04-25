import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AURC(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        r"""`Area Under the Risk-Coverage curve`_.

        The Area Under the Risk-Coverage curve (AURC) is the main metric for
        selective classification performance assessment. It evaluates the
        quality of uncertainty estimates by measuring the ability to
        discriminate between correct and incorrect predictions based on their
        rank (and not their values in contrast with calibration).

        As input to ``forward`` and ``update`` the metric accepts the following
            input:

        - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape
            ``(N, ...)`` containing probabilities for each observation.
        - ``target`` (:class:`~torch.Tensor`): An int tensor of shape
            ``(N, ...)`` containing ground-truth labels.

        As output to ``forward`` and ``compute`` the metric returns the
            following output:
        - ``aurc`` (:class:`~torch.Tensor`): A scalar tensor containing the
            area under the risk-coverage curve

        Args:
            kwargs: Additional keyword arguments.

        Reference:
            Geifman & El-Yaniv. "Selective classification for deep neural networks." In NeurIPS, 2017.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Store the scores and their associated errors for later computation.

        Args:
            probs (Tensor): The predicted probabilities of shape :math:`(N, C)`.
            targets (Tensor): The ground truth labels of shape :math:`(N,)`.
        """
        self.scores.append(-probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def partial_compute(self) -> tuple[Tensor, Tensor]:
        """Compute the error and optimal error rates for the RC curve.

        Returns:
            tuple[Tensor, Tensor]: The error rates and the optimal/oracle error
                rates.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        error_rates = _aurc_rejection_rate_compute(scores, errors)
        optimal_error_rates = _aurc_rejection_rate_compute(errors, errors)
        return error_rates.cpu(), optimal_error_rates.cpu()

    def compute(self) -> Tensor:
        """Compute the Area Under the Risk-Coverage curve (AURC).

        Returns:
            Tensor: The AURC.
        """
        error_rates, optimal_error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        x = np.arange(1, num_samples + 1) / num_samples
        y = (error_rates - optimal_error_rates).numpy()
        return torch.tensor([auc(x, y)])

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_oracle: bool = True,
        plot_value: bool = True,
        name: str | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the risk-cov. curve corresponding to the inputs passed to
        ``update``, and the oracle risk-cov. curve.

        Args:
            ax (Axes | None, optional): An matplotlib axis object. If provided
                will add plot to this axis. Defaults to None.
            plot_oracle (bool, optional): Whether to plot the oracle
                risk-cov. curve. Defaults to True.
            plot_value (bool, optional): Whether to print the AURC value on the
                plot. Defaults to True.
            name (str | None, optional): Name of the model. Defaults to None.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)

        # Computation of AUSEC
        error_rates, optimal_error_rates = self.partial_compute()
        num_errors = error_rates.size(0)
        x = np.arange(num_errors) / num_errors
        y = (error_rates - optimal_error_rates).numpy()
        aurc = auc(x, y)

        rejection_rates = (np.arange(num_errors) / num_errors) * 100

        ax.plot(
            100 - rejection_rates,
            error_rates * 100,
            label="Model" if name is None else name,
        )
        if plot_oracle:
            ax.plot(
                100 - rejection_rates,
                optimal_error_rates * 100,
                label="Oracle",
            )

        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_xlim(self.plot_lower_bound, self.plot_upper_bound)
        ax.set_ylim(self.plot_lower_bound, self.plot_upper_bound)
        ax.legend(loc="upper right")

        if plot_value:
            ax.text(
                0.02,
                0.95,
                f"AUSEC={aurc:.3%}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Coverage (%)", fontsize=16)
        ax.set_ylabel("Risk - Error Rate (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        return fig, ax


def _aurc_rejection_rate_compute(
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores (Tensor): uncertainty scores of shape :math:`(B,)`
        errors (Tensor): binary errors of shape :math:`(B,)`
    """
    num_samples = scores.size(0)
    errors = errors[scores.argsort()]
    cumulative_errors = errors.cumsum(dim=-1) / torch.arange(
        1, num_samples + 1, dtype=scores.dtype, device=scores.device
    )
    return cumulative_errors.flip(0)
