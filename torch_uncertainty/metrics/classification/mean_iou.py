from torch import Tensor
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class MeanIntersectionOverUnion(MulticlassStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs,
    ) -> None:
        r"""Computes Mean Intersection over Union (IoU) score.

        Args:
            num_classes (int): Integer specifying the number of classes.
            top_k (int, optional): Number of highest probability or logit score predictions
                considered to find the correct label. Only works when ``preds`` contain
                probabilities/logits. Defaults to ``1``.
            ignore_index (int | None, optional): Specifies a target value that is ignored and does
                not contribute to the metric calculation. Defaults to ``None``.
            validate_args (bool, optional): Bool indicating if input arguments and tensors should
                be validated for correctness. Set to ``False`` for faster computations. Defaults to
                ``True``.
            **kwargs: kwargs: Additional keyword arguments, see
                `Advanced metric settings <https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs>`_
                for more info.

        Shape:
            As input to ``forward`` and ``update`` the metric accepts the following input:

            - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(B, ...)`` or float tensor of shape ``(B, C, ..)``.
              If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
              probabilities/logits into an int tensor.
            - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(B, ...)``.

            As output to ``forward`` and ``compute`` the metric returns the following output:

            - ``mean_iou`` (:class:`~torch.Tensor`): The computed Mean Intersection over Union (IoU) score.
              A tensor containing a single float value.
        """
        super().__init__(
            num_classes,
            top_k,
            "macro",
            "global",
            ignore_index,
            validate_args,
            **kwargs,
        )

    def compute(self) -> Tensor:
        """Compute the Means Intersection over Union (MIoU) based on saved inputs."""
        tp, fp, _, fn = self._final_state()

        return _safe_divide(tp, tp + fp + fn, zero_division=float("nan")).nanmean()
