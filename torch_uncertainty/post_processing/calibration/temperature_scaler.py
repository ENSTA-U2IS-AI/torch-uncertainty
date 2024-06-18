from typing import Literal

import torch
from torch import Tensor, nn

from .scaler import Scaler


class TemperatureScaler(Scaler):
    def __init__(
        self,
        model: nn.Module | None = None,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Temperature scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            init_val (float, optional): Initial value for the temperature.
                Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to 100.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use
                for optimization. Defaults to None.

        Reference:
            Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.
        """
        super().__init__(model=model, lr=lr, max_iter=max_iter, device=device)

        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.set_temperature(init_val)

    def set_temperature(self, val: float) -> None:
        """Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temp = nn.Parameter(
            torch.ones(1, device=self.device) * val, requires_grad=True
        )

    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the prediction with the optimal temperature.

        Args:
            logits (Tensor): logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        return logits / self.temperature[0]

    @property
    def temperature(self) -> list:
        return [self.temp]
