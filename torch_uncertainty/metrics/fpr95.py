import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


def stable_cumsum(arr: ArrayLike, rtol: float = 1e-05, atol: float = 1e-08):
    """Uses high precision for cumsum and checks that the final value matches
    the sum.

    Args:
        arr (ArrayLike): The array to be cumulatively summed as flat.
        rtol (float, optional): Relative tolerance, see ``np.allclose``.
            Defaults to 1e-05.
        atol (float, optional): Absolute tolerance, see ``np.allclose``.
            Defaults to 1e-08.

    Returns:
        ArrayLike: The cumulatively summed array.

    Reference:
        From https://github.com/hendrycks/anomaly-seg.

    TODO: Check if necessary.
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(
        out[-1], expected, rtol=rtol, atol=atol
    ):  # coverage: ignore
        raise RuntimeError(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum"
        )
    return out


class FPR95(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    conf: list[Tensor]
    targets: list[Tensor]

    def __init__(self, pos_label: int, **kwargs) -> None:
        """The False Positive Rate at 95% Recall metric."""
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.add_state("conf", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `FPR95` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, conf: Tensor, target: Tensor) -> None:
        self.conf.append(conf)
        self.targets.append(target)

    def compute(self) -> Tensor:
        r"""Compute the actual False Positive Rate at 95% Recall.

        Returns:
            Tensor: The value of the FPR95.

        Reference:
            Inpired by https://github.com/hendrycks/anomaly-seg.
        """
        conf = dim_zero_cat(self.conf).cpu().numpy()
        targets = dim_zero_cat(self.targets).cpu().numpy()

        # out_labels is an array of 0s and 1s - 0 if IOD 1 if OOD
        out_labels = targets == self.pos_label

        in_scores = conf[np.logical_not(out_labels)]
        out_scores = conf[out_labels]

        # pos = OOD
        neg = np.array(in_scores[:]).reshape((-1, 1))
        pos = np.array(out_scores[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[: len(pos)] += 1

        # make labels a boolean vector, True if OOD
        labels = labels == self.pos_label

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(examples, kind="mergesort")[::-1]
        examples = examples[desc_score_indices]
        labels = labels[desc_score_indices]

        # examples typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(examples))[0]
        threshold_idxs = np.r_[distinct_value_indices, labels.shape[0] - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(labels)[threshold_idxs]
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

        thresholds = examples[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1]
        recall, fps, tps, thresholds = (
            np.r_[recall[sl], 1],
            np.r_[fps[sl], 0],
            np.r_[tps[sl], 0],
            thresholds[sl],
        )

        cutoff = np.argmin(np.abs(recall - 0.95))

        return torch.tensor(
            fps[cutoff] / (np.sum(np.logical_not(labels))), dtype=torch.float32
        )
