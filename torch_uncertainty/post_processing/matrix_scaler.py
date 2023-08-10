# fmt: off
from typing import Literal, Optional

import torch
from torch import nn

from .scaler import Scaler


# fmt: on
class MatrixScaler(Scaler):
    """
    Matrix scaling post-processing for calibrated probabilities.

    Args:
        num_classes (int): Number of classes.
        init_w (float, optional): Initial value for the weights.
            Defaults to 1.
        init_b (float, optional): Initial value for the bias.
            Defaults to 0.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations for the
            optimizer. Defaults to 100.
        device (Optional[Literal["cpu", "cuda"]], optional): Device to use
            for optimization. Defaults to None.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.

    """

    def __init__(
        self,
        num_classes: int,
        init_w: float = 1,
        init_b: float = 0,
        lr: float = 0.1,
        max_iter: int = 200,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, device=device)

        if not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer.")
        if num_classes <= 0:
            raise ValueError("The number of classes must be positive.")
        self.num_classes = num_classes

        self.set_temperature(init_w, init_b)

    def set_temperature(self, val_w: float, val_b: float) -> None:
        """
        Set the temperature to a fixed value.

        Args:
            val_w (float): Weight temperature value.
            val_b (float): Bias temperature value.
        """
        diag = torch.ones(self.num_classes, device=self.device)

        self.temp_w = nn.Parameter(
            diag * val_w,
            requires_grad=True,
        )
        self.temp_b = nn.Parameter(
            torch.ones(self.num_classes, device=self.device) * val_b,
            requires_grad=True,
        )

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale the logits with the optimal temperature.

        Args:
            logits (torch.Tensor): Logits to be scaled.

        Returns:
            torch.Tensor: Scaled logits.
        """
        return self.temp_w @ logits + self.temp_b

    @property
    def temperature(self) -> list:
        return [self.temp_w, self.temp_b]
