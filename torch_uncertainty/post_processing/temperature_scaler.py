# fmt: off
from typing import Literal, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .scaler import Scaler


# fmt: on
class TemperatureScaler(Scaler):
    """
    Temperature scaling post-processing for calibrated probabilities.

    Args:
        init_value (float, optional): Initial value for the temperature.
            Defaults to 1.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations for the
            optimizer. Defaults to 100.
        device (Optional[Literal["cpu", "cuda"]], optional): Device to use
            for optimization. Defaults to None.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.

    Note:
        Inspired by `<https://github.com/gpleiss/temperature_scaling>`_
    """

    def __init__(
        self,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, device=device)

        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.set_temperature(init_val)

    def set_temperature(self, val: float) -> None:
        """
        Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temp = nn.Parameter(
            torch.ones(1, device=self.device) * val, requires_grad=True
        )

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale the logits with the optimal temperature.

        Args:
            logits (torch.Tensor): Logits to be scaled.

        Returns:
            torch.Tensor: Scaled logits.
        """
        temperature = (
            self.temperature[0]
            .unsqueeze(1)
            .expand(logits.size(0), logits.size(1))
        )
        return logits / temperature

    def fit_predict(
        self, model: nn.Module, calib_loader: DataLoader, progress: bool = True
    ) -> torch.Tensor:
        self.fit(model, calib_loader, save_logits=True, progress=progress)
        calib_logits = self(self.logits)
        return calib_logits

    @property
    def temperature(self) -> list:
        return [self.temp]
