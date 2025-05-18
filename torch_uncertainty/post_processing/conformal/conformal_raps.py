import logging
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .abstract import Conformal


class ConformalClsRAPS(Conformal):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        score_type: str = "softmax",
        randomized: bool = True,
        penalty: float = 0.1,
        k_reg: int = 1,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformal prediction with RAPS scores.

        Args:
            alpha (float): The confidence level meaning we allow :math:`1-\alpha` error.
            model (nn.Module): Trained classification model. Defaults to ``None``.
            score_type (str): Type of score transformation. Only ``"softmax"`` is supported for now.
                Defaults to ``"softmax"``.
            randomized (bool): Whether to use randomized smoothing in RAPS. Defaults to ``True``.
            penalty (float): Regularization weight. Defaults to ``0.1``.
            k_reg (int): Rank threshold for regularization. Defaults to ``1``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.

        Reference:
            - TODO:
        """
        super().__init__(model=model)
        self.alpha = alpha
        self.score_type = score_type
        self.randomized = randomized
        self.penalty = penalty
        self.k_reg = k_reg
        self.q_hat = None
        self.device = device or "cpu"

        if self.score_type == "softmax":
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
        logits = self.model(inputs)
        return self.transform(logits)

    def _sort_sum(self, probs: Tensor):
        """Sort probabilities and compute cumulative sums."""
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs: Tensor, labels: Tensor) -> Tensor:
        """Compute RAPS score for the true label."""
        indices, ordered, cumsum = self._sort_sum(probs)
        batch_size = probs.shape[0]

        if self.randomized:
            noise = torch.rand(batch_size, device=probs.device)
        else:
            noise = torch.zeros(batch_size, device=probs.device)

        scores = torch.zeros(batch_size, device=probs.device)
        for i in range(batch_size):
            pos_tensor = (indices[i] == labels[i]).nonzero(as_tuple=False)
            if pos_tensor.numel() == 0:
                raise ValueError("True label not found.")
            pos = pos_tensor[0].item()

            reg = max(self.penalty * ((pos + 1) - self.k_reg), 0)
            scores[i] = cumsum[i, pos] - ordered[i, pos] * noise[i] + reg
        return scores

    def _calculate_all_labels(self, probs: Tensor) -> Tensor:
        """Compute RAPS scores for all labels."""
        indices, ordered, cumsum = self._sort_sum(probs)
        num_classes = probs.shape[1]
        noise = torch.rand_like(probs) if self.randomized else torch.zeros_like(probs)
        ranks = torch.arange(1, num_classes + 1, device=probs.device, dtype=torch.float)
        penalty_vector = self.penalty * (ranks - self.k_reg)
        penalty_vector = torch.clamp(penalty_vector, min=0)
        penalty_matrix = penalty_vector.unsqueeze(0).expand_as(ordered)
        modified_scores = cumsum - ordered * noise + penalty_matrix

        # Reorder scores back to original label order
        reordered_scores = torch.empty_like(modified_scores)
        reordered_scores.scatter_(dim=-1, index=indices, src=modified_scores)
        return reordered_scores

    @torch.no_grad()
    def fit(self, dataloader: DataLoader) -> None:
        """Calibrate the RAPS threshold q_hat on a calibration set."""
        self.model.eval()
        raps_scores = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            probs = self.model_forward(images)
            scores = self._calculate_single_label(probs, labels)
            raps_scores.append(scores)

        raps_scores = torch.cat(raps_scores)
        self.q_hat = torch.quantile(raps_scores, 1 - self.alpha)

    @torch.no_grad()
    def conformal(self, inputs: Tensor) -> Tensor:
        """Compute the prediction set for each input.

        Returns:
            Tensor: Uniform prediction over the predicted set size (B, C).
        """
        self.model.eval()
        probs = self.model_forward(inputs.to(self.device))
        pred_set = self._calculate_all_labels(probs) <= self.quantile
        confidence_score = 1 / pred_set.sum(dim=1, keepdim=True)
        return pred_set.float() * confidence_score

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise ValueError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat.detach()
