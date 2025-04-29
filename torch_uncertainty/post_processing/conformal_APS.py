from typing import Literal, Optional
import torch
from torch import nn, optim, Tensor
from .abstract import PostProcessing
from torch.utils.data import DataLoader
from .calibration import TemperatureScaler



class ConformalclassificationAPS(PostProcessing):
    def __init__(
            self,
            model: nn.Module,
            score_type: str = "softmax",
            randomized: bool = True,
            device: Optional[Literal["cpu", "cuda"]] = None,
            alpha: float = 0.1,
    ) -> None:
        """Conformal prediction with APS scores.

        Args:
            model (nn.Module): Trained classification model.
            score_type (str): Type of score transformation. Only 'softmax' is supported for now.
            randomized (bool): Whether to use randomized smoothing in APS.
            device (str, optional): 'cpu' or 'cuda'.
            alpha (float): Allowed miscoverage level.
        """
        super().__init__(model=model)
        self.model = model.to(device=device)
        self.randomized = randomized
        self.alpha = alpha
        self.device = device or "cpu"
        self.q_hat = None

        if score_type == "softmax":
            self.transform = lambda x: torch.softmax(x, dim=-1)
        else:
            raise NotImplementedError("Only softmax is supported for now.")

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the model and return transformed scores (softmax)."""
        logits = self.model(inputs)
        probs = self.transform(logits)
        return probs

    def _sort_sum(self, probs: Tensor):
        """Sort probabilities and compute cumulative sums."""
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs: Tensor, labels: Tensor):
        """Compute APS score for the true label."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)

        scores = torch.zeros(probs.shape[0], device=probs.device)
        for i in range(probs.shape[0]):
            pos = (indices[i] == labels[i]).nonzero(as_tuple=False)
            if pos.numel() == 0:
                raise ValueError("True label not found.")
            pos = pos[0].item()
            scores[i] = cumsum[i, pos] - U[i] * ordered[i, pos]
        return scores

    def _calculate_all_labels(self, probs: Tensor):
        """Compute APS scores for all labels."""
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)

        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def calibrate(self, dataloader: DataLoader) -> None:
        """Calibrate the APS threshold q_hat on a calibration set."""
        self.model.eval()
        aps_scores = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                probs = self.forward(images)
                scores = self._calculate_single_label(probs, labels)
                aps_scores.append(scores)

        aps_scores = torch.cat(aps_scores)
        self.q_hat = torch.quantile(aps_scores, 1 - self.alpha)
        print(f"APS calibration threshold (q_hat): {self.q_hat:.4f}")

    def fit(self, dataloader: DataLoader) -> None:
        """Alias for calibrate to match other API style."""
        self.calibrate(dataloader)

    def conformal(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the prediction set for each input."""
        if self.q_hat is None:
            raise ValueError("You must calibrate (fit) before calling conformal.")

        self.model.eval()
        with torch.no_grad():
            probs = self.forward(inputs)
            all_scores = self._calculate_all_labels(probs)

            pred_set = all_scores <= self.q_hat
            set_size = pred_set.sum(dim=1).float()

        return pred_set, set_size

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise ValueError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat.detach()












