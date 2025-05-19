from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import TemperatureScaler

from .abstract import Conformal


class ConformalClsTHR(Conformal):
    def __init__(
        self,
        alpha: float,
        model: nn.Module | None = None,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        r"""Conformal prediction post-processing for calibrated models.

        Args:
            alpha (float): The confidence level, meaning we allow :math:`1-\alpha` error.
            model (nn.Module, optional): Model to be calibrated. Defaults to ``None``.
            init_val (float, optional): Initial value for the temperature.
                Defaults to ``1``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to ``100``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.

        Reference:
            - `Least ambiguous set-valued classifiers with bounded error levels, Sadinle, M. et al., (2016) <https://arxiv.org/abs/1609.00451>`_.
        """
        super().__init__(model=model)
        self.alpha = alpha
        self.temperature_scaler = TemperatureScaler(
            model=model,
            init_val=init_val,
            lr=lr,
            max_iter=max_iter,
            device=device,
        )
        self.device = device or "cpu"
        self.q_hat = None  # Will be set after calibration

    def set_model(self, model: nn.Module) -> None:
        self.model = model.eval()
        self.temperature_scaler.set_model(model=model)

    def model_forward(self, inputs: Tensor) -> Tensor:
        """Apply temperature scaling."""
        return self.temperature_scaler(inputs.to(self.device))

    def fit(self, dataloader: DataLoader) -> None:
        self.temperature_scaler.fit(dataloader=dataloader)
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                scaled_logits = self.model_forward(images)
                logits_list.append(scaled_logits)
                labels_list.append(labels)

        probs = torch.cat(logits_list).softmax(-1)
        labels = torch.cat(labels_list).long()
        true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        scores = 1.0 - true_class_probs  # scores are (1 - true prob)
        self.q_hat = torch.quantile(scores, 1.0 - self.alpha)

    @torch.no_grad()
    def conformal(self, inputs: Tensor) -> Tensor:
        """Perform conformal prediction on the test set."""
        self.model.eval()
        probs = self.model_forward(inputs.to(self.device)).softmax(-1)
        pred_set = probs >= 1.0 - self.quantile
        top1 = torch.argmax(probs, dim=1, keepdim=True)
        pred_set.scatter_(1, top1, True)  # Always include top-1 class
        confidence_score = 1 / pred_set.sum(dim=1, keepdim=True)
        return pred_set.float() * confidence_score

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise ValueError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat

    @property
    def temperature(self) -> Tensor:
        """Get the temperature parameter."""
        return self.temperature_scaler.temperature[0].detach()
