import torch
from glest import GLEstimator as GLEstimatorBase
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class GLEstimator(GLEstimatorBase):
    def fit(self, probs: Tensor, targets: Tensor, features: Tensor):
        probs = probs.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        targets = (targets * 1).detach().cpu().numpy()
        if targets.ndim == 2:
            targets = targets.argmax(axis=1)
        self.classifier = probs
        return super().fit(features, targets)


class GroupingLoss(Metric):
    is_differentiable: bool = False
    higher_is_better: bool | None = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        **kwargs,
    ) -> None:
        r"""Metric to estimate the Grouping Loss.

        Args:
            num_classes (int): Number of classes
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)`
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`

            where :math:`B` is the batch size, :math:`C` is the number of classes
            and :math:`N` is the number of estimators.

        Note:
            If :attr:`probs` is a 3d tensor, then the metric computes the mean of
            the Brier score over the estimators ie. :math:`t = \frac{1}{N}
            \sum_{i=0}^{N-1} BrierScore(probs[:,i,:], target)`.

        Warning:
            Make sure that the probabilities in :attr:`probs` are normalized to sum
            to one.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.
        """
        super().__init__(**kwargs)
        self.estimator = GLEstimator(None)

        self.num_classes = num_classes

        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("features", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        rank_zero_warn(
            "Metric `GroupingLoss` will save all targets, predictions and features"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, probs: Tensor, target: Tensor, features: Tensor) -> None:
        """Accumulate the tensors for the estimation of the Grouping Loss.

        Args:
            probs (Tensor): A probability tensor of shape
                (batch, num_estimators, num_classes) or
                (batch, num_classes)
            target (Tensor): A tensor of ground truth labels of shape
                (batch, num_classes) or (batch)
            features (Tensor): A tensor of features of shape
                (batch, num_estimators, num_features) or (batch, num_features)
        """
        if probs.ndim == 2:
            max_probs = probs.max(-1)
            self.probs.append(max_probs.values)
            self.targets.append(target == max_probs.indices)
        elif probs.ndim == 3:
            max_probs = probs.mean(1).max(-1)
            self.probs.append(max_probs.values)
            self.targets.append(target == max_probs.indices)
        else:
            raise ValueError

        if features.ndim == 2:
            self.features.append(features)
        elif features.ndim == 3:
            self.features.append(features[:, 0, :])
        else:
            raise ValueError

        if probs.ndim not in [2, 3] or features.ndim not in [2, 3]:
            raise ValueError(
                f"Expected `probs` to be of shape (batch, num_classes) or "
                f"(batch, num_estimators, num_classes) but got {probs.shape}"
            )

    def compute(self) -> torch.Tensor:
        """Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            torch.Tensor: The final value(s) for the Brier score
        """
        probs = dim_zero_cat(self.probs)
        features = dim_zero_cat(self.features)
        targets = dim_zero_cat(self.targets)
        estimator = self.estimator.fit(probs, targets, features)
        return estimator.metrics("brier")["GL"]
