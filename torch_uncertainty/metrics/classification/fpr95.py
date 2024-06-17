import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class FPRx(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    conf: list[Tensor]
    targets: list[Tensor]

    def __init__(self, recall_level: float, pos_label: int, **kwargs) -> None:
        """The False Positive Rate at x% Recall metric.

        Args:
            recall_level (float): The recall level at which to compute the FPR.
            pos_label (int): The positive label.
            kwargs: Additional arguments to pass to the metric class.

        Reference:
            Inpired by https://github.com/hendrycks/anomaly-seg.
        """
        super().__init__(**kwargs)

        if recall_level < 0 or recall_level > 1:
            raise ValueError(
                f"Recall level must be between 0 and 1. Got {recall_level}."
            )
        self.recall_level = recall_level
        self.pos_label = pos_label
        self.add_state("conf", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            f"Metric `FPR{int(recall_level*100)}` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, conf: Tensor, target: Tensor) -> None:
        """Update the metric state.

        Args:
            conf (Tensor): The confidence scores.
            target (Tensor): The target labels.
        """
        self.conf.append(conf)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the actual False Positive Rate at x% Recall.

        Returns:
            Tensor: The value of the FPRx.
        """
        conf = dim_zero_cat(self.conf).cpu().numpy()
        targets = dim_zero_cat(self.targets).cpu().numpy()

        # out_labels is an array of 0s and 1s - 0 if IOD 1 if OOD
        out_labels = targets == self.pos_label

        in_scores = conf[np.logical_not(out_labels)]
        out_scores = conf[out_labels]

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
        tps = np.cumsum(labels)[threshold_idxs]
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

        cutoff = np.argmin(np.abs(recall - self.recall_level))

        return torch.tensor(
            fps[cutoff] / (np.sum(np.logical_not(labels))), dtype=torch.float32
        )


class FPR95(FPRx):
    def __init__(self, pos_label: int, **kwargs) -> None:
        """The False Positive Rate at 95% Recall metric.

        Args:
            recall_level (float): The recall level at which to compute the FPR.
            pos_label (int): The positive label.
            kwargs: Additional arguments to pass to the metric class.
        """
        super().__init__(recall_level=0.95, pos_label=pos_label, **kwargs)
