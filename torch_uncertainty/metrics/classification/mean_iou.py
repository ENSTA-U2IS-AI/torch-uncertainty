from typing import Literal

from torch import Tensor
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class MeanIntersectionOverUnion(MulticlassStatScores):
    """Compute the MeanIntersection over Union (IoU) score."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            top_k,
            "macro",
            multidim_average,
            ignore_index,
            validate_args,
            **kwargs,
        )

    def compute(self) -> Tensor:
        """Compute the Means Intersection over Union (MIoU) based on saved inputs."""
        tp, fp, _, fn = self._final_state()
        return _safe_divide(tp, tp + fp + fn).mean()
