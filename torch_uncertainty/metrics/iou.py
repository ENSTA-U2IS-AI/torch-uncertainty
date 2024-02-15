from einops import rearrange
from torch import Tensor
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class IntersectionOverUnion(MulticlassStatScores):
    """Compute the Intersection over Union (IoU) score."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds (Tensor): prediction images of shape :math:`(B, H, W)` or
                :math:`(B, C, H, W)`.
            target (Tensor): target images of shape :math:`(B, H, W)`.
        """
        if preds.ndim == 3:
            preds = preds.flatten()
        if preds.ndim == 4:
            preds = rearrange(preds, "b c h w -> (b h w) c")

        target = target.flatten()

        super().update(preds, target)

    def compute(self) -> Tensor:
        """Compute the Intersection over Union (IoU) based on saved inputs."""
        tp, fp, _, fn = self._final_state()
        return _safe_divide(tp, tp + fp + fn)
