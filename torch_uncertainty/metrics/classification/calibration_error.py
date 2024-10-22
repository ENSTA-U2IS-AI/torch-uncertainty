from typing import Any, Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
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

from .adaptive_calibration_error import AdaptiveCalibrationError


def _reliability_diagram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    bin_sizes: np.ndarray,
    bins: np.ndarray,
    title: str = "Reliability Diagram",
    xlabel: str = "Confidence",
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
        positions,
        np.abs(accuracies - confidences),
        bottom=np.minimum(accuracies, confidences),
        width=widths,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions,
        0,
        bottom=accuracies,
        width=widths,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=3,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    gaps = np.abs(accuracies - confidences)
    ece = (np.sum(gaps * bin_sizes) / np.sum(bin_sizes)) * 100

    ax.text(
        0.98,
        0.02,
        f"ECE={ece:.03}%",
        color="black",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _confidence_histogram_subplot(
    ax,
    accuracies: np.ndarray,
    confidences: np.ndarray,
    title="Examples per bin",
    xlabel="Top-class Confidence (%)",
    ylabel="Density",
) -> None:
    sns.kdeplot(
        confidences,
        linewidth=2,
        ax=ax,
        fill=True,
        alpha=0.5,
    )

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)

    acc_plt = ax.axvline(
        x=avg_acc,
        ls="solid",
        lw=3,
        c="black",
        label="Accuracy",
    )
    conf_plt = ax.axvline(
        x=avg_conf,
        ls="dotted",
        lw=3,
        c="#444",
        label="Avg. confidence",
    )
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
    dpi=72,
) -> Figure:
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
        xlabel="",
    )

    # confidence histogram subplot
    _confidence_histogram_subplot(ax[1], accuracies, confidences, title="")

    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int32)
    ax[1].yaxis.set_major_locator(mticker.FixedLocator(new_ticks))
    ax[1].set_yticklabels(new_ticks)

    return fig


def custom_plot(self) -> plt.Figure:
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
