from torch import Tensor
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class MeanIntersectionOverUnion(MulticlassStatScores):
    """Compute the MeanIntersection over Union (IoU) score."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Compute the Means Intersection over Union (MIoU) based on saved inputs."""
        tp, fp, _, fn = self._final_state()
        return _safe_divide(tp, tp + fp + fn).mean()
