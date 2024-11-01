from importlib import util

import torch

if util.find_spec("glest"):
    from glest import GLEstimator as GLEstimatorBase

    glest_installed = True
else:  # coverage: ignore
    glest_installed = False
    GLEstimatorBase = object

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class GLEstimator(GLEstimatorBase):
    def fit(
        self, probs: Tensor, targets: Tensor, features: Tensor
    ) -> "GLEstimator":
        probs = probs.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        targets = (targets * 1).detach().cpu().numpy()
        self.classifier = probs
        return super().fit(features, targets)


class GroupingLoss(Metric):
    is_differentiable: bool = False
    higher_is_better: bool | None = False
    full_state_update: bool = False

    def __init__(
        self,
        **kwargs,
    ) -> None:
        r"""Metric to estimate the Top-label Grouping Loss.

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)`
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`
            - :attr:`features`: :math:`(B, F)` or :math:`(B, N, F)`

            where :math:`B` is the batch size, :math:`C` is the number of classes
            and :math:`N` is the number of estimators.

        Warning:
            Make sure that the probabilities in :attr:`probs` are normalized to sum
            to one.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.

        Reference:
            Perez-Lebel, Alexandre, Le Morvan, Marine and Varoquaux, Gaël.
            Beyond calibration: estimating the grouping loss of modern neural
            networks. In ICLR 2023.
        """
        super().__init__(**kwargs)
        if not glest_installed:  # coverage: ignore
            raise ImportError(
                "The glest library is not installed. Please install"
                "torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )

        self.estimator = GLEstimator(None)

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
            probs (Tensor): A probability tensor of shape (batch, num_classes),
                (batch, num_estimators, num_classes), or (batch) if binary
                classification
            target (Tensor): A tensor of ground truth labels of shape
                (batch, num_classes) or (batch)
            features (Tensor): A tensor of features of shape
                (batch, num_estimators, num_features) or (batch, num_features)
        """
        if target.ndim == 2:
            target = target.argmax(dim=-1)
        elif target.ndim != 1:
            raise ValueError(
                "Expected `target` to be of shape (batch) or (batch, num_classes) "
                f"but got {target.shape}."
            )

        if probs.ndim == 1:
            self.probs.append(probs)
            self.targets.append(target == (probs > 0.5).int())
        elif probs.ndim == 2:
            max_probs = probs.max(-1)
            self.probs.append(max_probs.values)
            self.targets.append(target == max_probs.indices)
        elif probs.ndim == 3:
            max_probs = probs.mean(1).max(-1)
            self.probs.append(max_probs.values)
            self.targets.append(target == max_probs.indices)
        else:
            raise ValueError(
                "Expected `probs` to be of shape (batch, num_classes) or "
                "(batch, num_estimators, num_classes) or (batch) "
                f"but got {probs.shape}."
            )

        if features.ndim == 2:
            self.features.append(features)
        elif features.ndim == 3:
            self.features.append(features[:, 0, :])
        else:
            raise ValueError(
                "Expected `features` to be of shape (batch, num_features) or "
                "(batch, num_estimators, num_features) but got "
                f"{features.shape}."
            )

    def compute(self) -> Tensor:
        """Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            torch.Tensor: The final value(s) for the Brier score
        """
        probs = dim_zero_cat(self.probs)
        features = dim_zero_cat(self.features)
        targets = dim_zero_cat(self.targets)
        estimator = self.estimator.fit(probs, targets, features)
        return estimator.metrics("brier")["GL"]
