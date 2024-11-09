from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from torchmetrics.classification.calibration_error import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)
from torchmetrics.functional.classification.calibration_error import (
    _binning_bucketize,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.plot import _PLOT_OUT_TYPE

from .adaptive_calibration_error import AdaptiveCalibrationError


def _reliability_diagram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bins: np.ndarray,
    title: str = "Reliability Diagram",
    xlabel: str = "Top-class Confidence (%)",
    ylabel: str = "Success Rate (%)",
) -> None:
    widths = 1.0 / len(bin_sizes)
    positions = bins + widths / 2.0
    alphas = 0.2 + 0.8 * bin_sizes

    colors = np.zeros((len(bin_sizes), 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = alphas

    gap_plt = ax.bar(
        positions * 100,
        np.abs(accuracies - confidences) * 100,
        bottom=np.minimum(accuracies, confidences) * 100,
        width=widths * 100,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions * 100,
        0,
        bottom=accuracies * 100,
        width=widths * 100,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=2,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray")

    gaps = np.abs(accuracies - confidences)
    ece = np.sum(gaps * bin_sizes) / np.sum(bin_sizes)

    ax.text(
        0.98,
        0.02,
        f"ECE={ece:.02%}",
        color="black",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    title="Examples per bin",
    xlabel="Top-class Confidence (%)",
    ylabel="Density (%)",
) -> None:
    sns.kdeplot(
        confidences * 100,
        linewidth=2,
        ax=ax,
        fill=True,
        alpha=0.5,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)

    acc_plt = ax.axvline(
        x=avg_acc * 100,
        ls="solid",
        lw=2,
        c="black",
        label="Accuracy",
    )
    conf_plt = ax.axvline(
        x=avg_conf * 100,
        ls="dotted",
        lw=2,
        c="#444",
        label="Avg. confidence",
    )
    ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax.legend(handles=[acc_plt, conf_plt], loc="upper left")


def reliability_chart(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bins: np.ndarray,
    title="Reliability Diagram",
    figsize=(6, 6),
    dpi=150,
) -> _PLOT_OUT_TYPE:
    """Builds Reliability Diagram
    `Source <https://github.com/hollance/reliability-diagrams>`_.
    """
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    # reliability diagram subplot
    _reliability_diagram_subplot(
        ax[0],
        bin_accuracies,
        bin_confidences,
        bin_sizes,
        bins,
        title=title,
    )

    # confidence histogram subplot
    _confidence_histogram_subplot(ax[1], accuracies, confidences, title="")
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    return fig, ax


def custom_plot(self) -> _PLOT_OUT_TYPE:
    confidences = dim_zero_cat(self.confidences)
    accuracies = dim_zero_cat(self.accuracies)

    bin_boundaries = torch.linspace(
        0,
        1,
        self.n_bins + 1,
        dtype=torch.float,
        device=confidences.device,
    )

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(
            confidences, accuracies, bin_boundaries
        )

    np_acc_bin = acc_bin.cpu().numpy()
    np_conf_bin = conf_bin.cpu().numpy()
    np_prop_bin = prop_bin.cpu().numpy()
    np_bin_boundaries = bin_boundaries.cpu().numpy()

    return reliability_chart(
        accuracies=accuracies.cpu().numpy(),
        confidences=confidences.cpu().numpy(),
        bin_accuracies=np_acc_bin,
        bin_confidences=np_conf_bin,
        bin_sizes=np_prop_bin,
        bins=np_bin_boundaries,
    )


# overwrite the plot method of the original metrics
BinaryCalibrationError.plot = custom_plot
MulticlassCalibrationError.plot = custom_plot


class CalibrationError:
    r"""Top-label Calibration Error.

    See
    `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
    for details. Our version of the metric is a wrapper around the original
    metric providing a plotting functionality.

    Reference:
        Naeini et al. "Obtaining well calibrated probabilities using Bayesian
            binning." In AAAI, 2015.
    """

    def __new__(  # type: ignore[misc]
        cls,
        task: Literal["binary", "multiclass"],
        adaptive: bool = False,
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        if kwargs.get("n_bins") is not None:
            raise ValueError("`n_bins` does not exist, use `num_bins`.")
        if adaptive:
            return AdaptiveCalibrationError(
                task=task,
                num_bins=num_bins,
                norm=norm,
                num_classes=num_classes,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            )
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": num_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        #  task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassCalibrationError(num_classes, **kwargs)
