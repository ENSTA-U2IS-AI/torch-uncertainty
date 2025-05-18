import logging
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import Conformal


class ConformalClsAPS(Conformal):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        score_type: str = "softmax",
        randomized: bool = True,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformal prediction with APS scores.

        Args:
            alpha (float): The confidence level meaning we allow :math:`1-\alpha` error.
            model (nn.Module): Trained classification model. Defaults to ``None``.
            score_type (str): Type of score transformation. Only ``"softmax"`` is supported for now.
                Defaults to ``"softmax"``.
            randomized (bool): Whether to use randomized smoothing in APS. Defaults to ``True``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.

        Reference:
            - TODO:
        """
        super().__init__(model=model)
        self.randomized = randomized
        self.alpha = alpha
        self.device = device or "cpu"
        self.q_hat = None

        if score_type == "softmax":
            self.transform = partial(torch.softmax, dim=-1)
        else:
            raise NotImplementedError("Only softmax is supported for now.")

    def model_forward(self, inputs: Tensor) -> Tensor:
        """Apply the model and return transformed scores (softmax)."""
        if self.model is None or isinstance(self.model, nn.Identity):
            logging.warning(
                "model is None. Fitting post_processing method on the dataloader's data directly."
            )
            self.model = nn.Identity()
        logits = self.model(inputs.to(self.device))
        return self.transform(logits)

    def _sort_sum(self, probs: Tensor):
        """Sort probabilities and compute cumulative sums."""
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs: Tensor, labels: Tensor):
        """Compute APS score for the true label."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            u = torch.rand(indices.shape[0], device=probs.device)
        else:
            u = torch.zeros(indices.shape[0], device=probs.device)

        scores = torch.zeros(probs.shape[0], device=probs.device)
        for i in range(probs.shape[0]):
            pos = (indices[i] == labels[i]).nonzero(as_tuple=False)
            if pos.numel() == 0:
                raise ValueError("True label not found.")
            pos = pos[0].item()
            scores[i] = cumsum[i, pos] - u[i] * ordered[i, pos]
        return scores

    def _calculate_all_labels(self, probs: Tensor):
        """Compute APS scores for all labels."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            u = torch.rand(probs.shape, device=probs.device)
        else:
            u = torch.zeros_like(probs, device=probs.device)
        ordered_scores = cumsum - ordered * u
        sorted_indices = torch.sort(indices, descending=False, dim=-1).indices
        return ordered_scores.gather(dim=-1, index=sorted_indices)

    @torch.no_grad()
    def fit(self, dataloader: DataLoader) -> None:
        """Calibrate the APS threshold q_hat on a calibration set."""
        self.model.eval()
        aps_scores = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            probs = self.model_forward(images)
            scores = self._calculate_single_label(probs, labels)
            aps_scores.append(scores)

        aps_scores = torch.cat(aps_scores)
        self.q_hat = torch.quantile(aps_scores, 1 - self.alpha)

    @torch.no_grad()
    def conformal(self, inputs: Tensor) -> Tensor:
        """Compute the prediction set for each input."""
        self.model.eval()
        probs = self.model_forward(inputs.to(self.device))
        all_scores = self._calculate_all_labels(probs)
        pred_set = all_scores <= self.quantile
        confidence_score = 1 / pred_set.sum(dim=1, keepdim=True)
        return pred_set.float() * confidence_score

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise ValueError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat.detach()
