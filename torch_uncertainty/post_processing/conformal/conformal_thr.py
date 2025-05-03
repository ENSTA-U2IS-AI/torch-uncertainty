from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import TemperatureScaler

from .abstract import Conformal


class ConformalClsTHR(Conformal):
    def __init__(
        self,
        model: nn.Module,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
        alpha: float = 0.1,
    ) -> None:
        r"""Conformal prediction post-processing for calibrated models.

        Args:
            model (nn.Module): Model to be calibrated.
            init_val (float, optional): Initial value for the temperature.
                Defaults to ``1``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to ``100``.
            device (Literal["cpu", "cuda"] | torch.device | None, optional): device.
                Defaults to ``None``.
            alpha (float): The confidence level meaning we allow :math:`1-\alpha` error. Defaults
                to ``0.1``.

        Reference:
            - `Least ambiguous set-valued classifiers with bounded error levels, Sadinle, M. et al., (2016) <https://arxiv.org/abs/1609.00451>`_.
        """
        super().__init__(model=model)
        self.model = model.to(device=device)
        self.init_val = init_val
        self.lr = lr
        self.max_iter = max_iter
        self.device = device or "cpu"
        self.temp = None  # Will be set after calibration
        self.q_hat = None  # Will be set after calibration
        self.alpha = alpha

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply temperature scaling."""
        logits = self.model(inputs)
        return logits / self.temp

    def fit_temperature(self, dataloader: DataLoader) -> None:
        # Fit the scaler on the calibration dataset
        scaled_model = TemperatureScaler(
            model=self.model,
            init_val=self.init_val,
            lr=self.lr,
            max_iter=self.max_iter,
            device=self.device,
        )
        scaled_model.fit(dataloader=dataloader)
        self.temp = scaled_model.temperature[0].item()

    def fit(self, dataloader: DataLoader) -> None:
        self.fit_temperature(dataloader=dataloader)
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                scaled_logits = self.model(images) / self.temp
                logits_list.append(scaled_logits)
                labels_list.append(labels)

            scaled_logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            probs = torch.softmax(scaled_logits, dim=1)
            true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            scores = 1.0 - true_class_probs  # scores are (1 - true prob)
            # Quantile
            self.q_hat = torch.quantile(scores, 1.0 - self.alpha)

    def conformal(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Perform conformal prediction on the test set."""
        if self.q_hat is None:
            raise ValueError("You must calibrate and estimate the qhat first by calling `.fit()`.")

        self.model.eval()
        with torch.no_grad():
            scaled_logits = self.model(inputs) / self.temp
            probs = torch.softmax(scaled_logits, dim=1)
            pred_set = probs >= (1.0 - self.q_hat)
            top1 = torch.argmax(probs, dim=1, keepdim=True)
            pred_set.scatter_(1, top1, True)  # Always include top-1 class

            confidence_score = pred_set.sum(dim=1).float()

            return (
                pred_set,
                confidence_score,
            )

    @property
    def quantile(self) -> Tensor:
        if self.q_hat is None:
            raise ValueError("Quantile q_hat is not set. Run `.fit()` first.")
        return self.q_hat.detach()
