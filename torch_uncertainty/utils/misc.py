import copy
import csv
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.utils.data import Dataset, random_split


def csv_writer(path: Path, dic: dict) -> None:
    """Write a dictionary to a csv file.

    Args:
        path (Path): Path to the csv file.
        dic (dict): Dictionary to write.
    """
    if path.is_file():  # Check that the file already exists
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"
    # Write dic
    with path.open(rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(dic.keys())
        writer.writerow([f"{elem:.4f}" for elem in dic.values()])


def plot_hist(
    conf: list[torch.Tensor],
    bins: int = 20,
    title: str = "Histogram with 'auto' bins",
    dpi: int = 60,
) -> tuple[Figure, Axes]:
    """Plot a confidence histogram.

    Args:
        conf (Any): The confidence values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str, optional): The title of the plot. Defaults to "Histogram
            with 'auto' bins".
        dpi (int, optional): The dpi of the plot. Defaults to 60.

    Returns:
        Tuple[Figure, Axes]: The figure and axes of the plot.
    """
    plt.rc("axes", axisbelow=True)
    fig, ax = plt.subplots(1, figsize=(7, 5), dpi=dpi)
    for i in [1, 0]:
        ax.hist(
            conf[i],
            bins=bins,
            density=True,
            label=["In-distribution", "Out-of-Distribution"][i],
            alpha=0.4,
            linewidth=1,
            edgecolor=["#0d559f", "#d45f00"][i],
            color=["#1f77b4", "#ff7f0e"][i],
        )

    ax.set_title(title)
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    plt.legend()
    fig.tight_layout()
    return fig, ax


def create_train_val_split(
    dataset: Dataset,
    val_split_rate: float,
    val_transforms: Callable | None = None,
) -> tuple[Dataset, Dataset]:
    train, val = random_split(dataset, [1 - val_split_rate, val_split_rate])
    val = copy.deepcopy(val)
    val.dataset.transform = val_transforms
    return train, val
