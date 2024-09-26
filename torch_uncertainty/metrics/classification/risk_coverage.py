import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _auc_compute
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE


class AURC(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, **kwargs) -> None:
        r"""Area Under the Risk-Coverage curve.

        The Area Under the Risk-Coverage curve (AURC) is the main metric for
        Selective Classification (SC) performance assessment. It evaluates the
        quality of uncertainty estimates by measuring the ability to
        discriminate between correct and incorrect predictions based on their
        rank (and not their values in contrast with calibration).

        As input to ``forward`` and ``update`` the metric accepts the following input:

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
            Geifman & El-Yaniv. "Selective classification for deep neural
                networks." In NeurIPS, 2017.
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
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def partial_compute(self) -> Tensor:
        """Compute the error and optimal error rates for the RC curve.

        Returns:
            Tensor: The error rates and the optimal/oracle error
                rates.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        return _aurc_rejection_rate_compute(scores, errors)

    def compute(self) -> Tensor:
        """Compute the Area Under the Risk-Coverage curve (AURC).

        Normalize the AURC as if its support was between 0 and 1. This has an
        impact on the AURC when the number of samples is small.

        Returns:
            Tensor: The AURC.
        """
        error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        if num_samples < 2:
            return torch.tensor([float("nan")], device=self.device)
        cov = torch.arange(1, num_samples + 1, device=self.device) / num_samples
        return _auc_compute(cov, error_rates) / (1 - 1 / num_samples)

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_value: bool = True,
        name: str | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the risk-cov. curve corresponding to the inputs passed to
        ``update``.

        Args:
            ax (Axes | None, optional): An matplotlib axis object. If provided
                will add plot to this axis. Defaults to None.
            plot_value (bool, optional): Whether to print the AURC value on the
                plot. Defaults to True.
            name (str | None, optional): Name of the model. Defaults to None.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)

        # Computation of AURC
        error_rates = self.partial_compute().cpu().flip(0)
        num_samples = error_rates.size(0)

        x = torch.arange(num_samples) / num_samples
        aurc = _auc_compute(x, error_rates).cpu().item()

        # reduce plot size
        plot_xs = np.arange(0.01, 100 + 0.01, 0.01)
        xs = np.arange(start=1, stop=num_samples + 1) / num_samples

        rejection_rates = np.interp(plot_xs, xs, x * 100)
        error_rates = np.interp(plot_xs, xs, error_rates)

        # plot
        ax.plot(
            100 - rejection_rates,
            error_rates * 100,
            label="Model" if name is None else name,
        )

        if plot_value:
            ax.text(
                0.02,
                0.95,
                f"AURC={aurc:.2%}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Coverage (%)", fontsize=16)
        ax.set_ylabel("Risk - Error Rate (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, min(100, np.ceil(error_rates.max() * 100)))
        ax.legend(loc="upper right")
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
    errors = errors[scores.argsort(descending=True)]
    return errors.cumsum(dim=-1) / torch.arange(
        1, scores.size(0) + 1, dtype=scores.dtype, device=scores.device
    )


class AUGRC(AURC):
    """Area Under the Generalized Risk-Coverage curve.

    The Area Under the Generalized Risk-Coverage curve (AUGRC) for
    Selective Classification (SC) performance assessment. It avoids putting too much
    weight on the most confident samples.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape
        ``(N, ...)`` containing probabilities for each observation.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape
        ``(N, ...)`` containing ground-truth labels.

    As output to ``forward`` and ``compute`` the metric returns the
        following output:

    - ``augrc`` (:class:`~torch.Tensor`): A scalar tensor containing the
        area under the risk-coverage curve

    Args:
        kwargs: Additional keyword arguments.

    Reference:
        Traub et al. Overcoming Common Flaws in the Evaluation of Selective
        Classification Systems. ArXiv.
    """

    def compute(self) -> Tensor:
        """Compute the Area Under the Generalized Risk-Coverage curve (AUGRC).

        Normalize the AUGRC as if its support was between 0 and 1. This has an
        impact on the AUGRC when the number of samples is small.

        Returns:
            Tensor: The AUGRC.
        """
        error_rates = self.partial_compute()
        num_samples = error_rates.size(0)
        if num_samples < 2:
            return torch.tensor([float("nan")], device=self.device)
        cov = torch.arange(1, num_samples + 1, device=self.device) / num_samples
        return _auc_compute(cov, error_rates * cov) / (1 - 1 / num_samples)

    def plot(
        self,
        ax: _AX_TYPE | None = None,
        plot_value: bool = True,
        name: str | None = None,
    ) -> tuple[plt.Figure | None, plt.Axes]:
        """Plot the generalized risk-cov. curve corresponding to the inputs passed to
        ``update``.

        Args:
            ax (Axes | None, optional): An matplotlib axis object. If provided
                will add plot to this axis. Defaults to None.
            plot_value (bool, optional): Whether to print the AURC value on the
                plot. Defaults to True.
            name (str | None, optional): Name of the model. Defaults to None.

        Returns:
            tuple[[Figure | None], Axes]: Figure object and Axes object
        """
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)

        # Computation of AUGRC
        error_rates = self.partial_compute().cpu().flip(0)
        num_samples = error_rates.size(0)
        cov = torch.arange(num_samples) / num_samples

        augrc = _auc_compute(cov, error_rates * cov).cpu().item()

        # reduce plot size
        plot_covs = np.arange(0.01, 100 + 0.01, 0.01)
        covs = np.arange(start=1, stop=num_samples + 1) / num_samples

        rejection_rates = np.interp(plot_covs, covs, cov * 100)
        error_rates = np.interp(plot_covs, covs, error_rates * covs[::-1] * 100)

        # plot
        ax.plot(
            100 - rejection_rates,
            error_rates,
            label="Model" if name is None else name,
        )

        if plot_value:
            ax.text(
                0.02,
                0.95,
                f"AUGRC={augrc:.2%}",
                color="black",
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Coverage (%)", fontsize=16)
        ax.set_ylabel("Generalized Risk (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, min(100, np.ceil(error_rates.max() * 100)))
        ax.legend(loc="upper right")
        return fig, ax


class CovAtxRisk(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, risk_threshold: float, **kwargs) -> None:
        r"""Coverage at x Risk.

        If there are multiple coverage values corresponding to the given risk,
        i.e., the risk(coverage) is not monotonic, the coverage at x risk is
        the maximum coverage value corresponding to the given risk. If no
        there is no coverage value corresponding to the given risk, return
        float("nan").

        Args:
            risk_threshold (float): The risk threshold at which to compute the
                coverage.
            kwargs: Additional arguments to pass to the metric class.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        _risk_coverage_checks(risk_threshold)
        self.risk_threshold = risk_threshold

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Store the scores and their associated errors for later computation.

        Args:
            probs (Tensor): The predicted probabilities of shape :math:`(N, C)`.
            targets (Tensor): The ground truth labels of shape :math:`(N,)`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def compute(self) -> Tensor:
        """Compute the coverage at x Risk.

        Returns:
            Tensor: The coverage at x risk.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        num_samples = scores.size(0)
        if num_samples < 1:
            return torch.tensor([float("nan")], device=self.device)
        error_rates = _aurc_rejection_rate_compute(scores, errors)
        admissible_risks = (error_rates > self.risk_threshold) * 1
        max_cov_at_risk = admissible_risks.flip(0).argmin()

        # check if max_cov_at_risk is really admissible, if not return nan
        risk = admissible_risks[max_cov_at_risk]
        if risk > self.risk_threshold:
            return torch.tensor([float("nan")], device=self.device)
        return 1 - max_cov_at_risk / num_samples


class CovAt5Risk(CovAtxRisk):
    def __init__(self, **kwargs) -> None:
        r"""Coverage at 5% Risk.

        If there are multiple coverage values corresponding to 5% risk, the
        coverage at 5% risk is the maximum coverage value corresponding to 5%
        risk. If no there is no coverage value corresponding to the given risk,
        this metric returns float("nan").
        """
        super().__init__(risk_threshold=0.05, **kwargs)


class RiskAtxCov(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    scores: list[Tensor]
    errors: list[Tensor]

    def __init__(self, cov_threshold: float, **kwargs) -> None:
        r"""Risk at given Coverage.

        Args:
            cov_threshold (float): The coverage threshold at which to compute
                the risk.
            kwargs: Additional arguments to pass to the metric class.
        """
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        _risk_coverage_checks(cov_threshold)
        self.cov_threshold = cov_threshold

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Store the scores and their associated errors for later computation.

        Args:
            probs (Tensor): The predicted probabilities of shape :math:`(N, C)`.
            targets (Tensor): The ground truth labels of shape :math:`(N,)`.
        """
        if probs.ndim == 1:
            probs = torch.stack([1 - probs, probs], dim=-1)
        self.scores.append(probs.max(-1).values)
        self.errors.append((probs.argmax(-1) != targets) * 1.0)

    def compute(self) -> Tensor:
        """Compute the risk at given coverage.

        Returns:
            Tensor: The risk at given coverage.
        """
        scores = dim_zero_cat(self.scores)
        errors = dim_zero_cat(self.errors)
        error_rates = _aurc_rejection_rate_compute(scores, errors)
        return error_rates[math.ceil(scores.size(0) * self.cov_threshold) - 1]


class RiskAt80Cov(RiskAtxCov):
    def __init__(self, **kwargs) -> None:
        r"""Risk at 80% Coverage."""
        super().__init__(cov_threshold=0.8, **kwargs)


def _risk_coverage_checks(threshold: float) -> None:
    if not isinstance(threshold, float):
        raise TypeError(
            f"Expected threshold to be of type float, but got {type(threshold)}"
        )
    if threshold < 0 or threshold > 1:
        raise ValueError(
            f"Threshold should be in the range [0, 1], but got {threshold}."
        )
