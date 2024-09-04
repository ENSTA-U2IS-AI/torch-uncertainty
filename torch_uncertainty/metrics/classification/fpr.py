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
            Improved from https://github.com/hendrycks/anomaly-seg and
            translated to torch.
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
            target (Tensor): The target labels, 0 if ID, 1 if OOD.
        """
        self.conf.append(conf)
        self.targets.append(target)

    def compute(self) -> Tensor:
        """Compute the False Positive Rate at x% Recall.

        Returns:
            Tensor: The value of the FPRx.
        """
        conf = dim_zero_cat(self.conf)
        targets = dim_zero_cat(self.targets)

        # map examples and labels to OOD first
        indx = torch.argsort(targets, descending=True)
        examples = conf[indx]
        labels = torch.zeros_like(targets, dtype=torch.bool, device=self.device)
        labels[: torch.count_nonzero(targets)] = True

        # sort examples and labels by decreasing confidence
        desc_scores_indx = torch.argsort(examples, descending=True)
        examples = examples[desc_scores_indx]
        labels = labels[desc_scores_indx]

        # Get the indices of the distinct values
        distinct_value_indices = torch.where(torch.diff(examples))[0]
        threshold_idxs = torch.cat(
            [
                distinct_value_indices,
                torch.tensor(
                    [labels.shape[0] - 1], dtype=torch.long, device=self.device
                ),
            ]
        )

        # accumulate the true positives with decreasing threshold
        true_pos = torch.cumsum(labels, dim=0)[threshold_idxs]
        false_pos = (
            1 + threshold_idxs - true_pos
        )  # add one because of zero-based indexing

        # check that there is at least one OOD example
        if true_pos[-1] == 0:
            return torch.tensor([torch.nan], device=self.device)

        recall = true_pos / true_pos[-1]

        last_ind = torch.searchsorted(true_pos, true_pos[-1])
        recall = torch.cat(
            [
                recall[: last_ind + 1].flip(0),
                torch.tensor([1.0], device=self.device),
            ]
        )
        false_pos = torch.cat(
            [
                false_pos[: last_ind + 1].flip(0),
                torch.tensor([0.0], device=self.device),
            ]
        )
        cutoff = torch.argmin(torch.abs(recall - self.recall_level))
        return false_pos[cutoff] / (~labels).sum()


class FPR95(FPRx):
    def __init__(self, pos_label: int, **kwargs) -> None:
        """The False Positive Rate at 95% Recall metric.

        Args:
            pos_label (int): The positive label.
            kwargs: Additional arguments to pass to the metric class.
        """
        super().__init__(recall_level=0.95, pos_label=pos_label, **kwargs)
