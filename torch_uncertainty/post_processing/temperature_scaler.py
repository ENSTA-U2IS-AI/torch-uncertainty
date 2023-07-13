# fmt: off
from typing import Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# fmt: on
class TemperatureScaler(nn.Module):
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

    trained = False

    def __init__(
        self,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        super().__init__()
        self.device = device
        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.temperature = nn.Parameter(
            torch.ones(1, device=device) * init_val, requires_grad=True
        )
        self.criterion = nn.CrossEntropyLoss()

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def set_temperature(self, val: float) -> None:
        """
        Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if val <= 0:
            raise ValueError("Temperature value must be positive.")

        self.temperature = nn.Parameter(
            torch.ones(1, device=self.device) * val, requires_grad=True
        )

    def fit(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        save_logits: bool = False,
        progress: bool = True,
    ) -> "TemperatureScaler":
        """
        Fit the temperature to the validation data.

        Args:
            model (nn.Module): Model to calibrate.
            calib_loader (DataLoader): Calibration dataloader.
            save_logits (bool, optional): Whether to save the logits and
                labels. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.

        Returns:
            TemperatureScaler: Calibrated scaler.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(calib_loader, disable=not progress):
                input = input.to(self.device)
                logits = model(input)
                logits_list.append(logits)
                labels_list.append(label)
        logits = torch.cat(logits_list).detach().to(self.device)
        labels = torch.cat(labels_list).detach().to(self.device)

        optimizer = optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )

        def eval() -> float:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        self.trained = True
        if save_logits:
            self.logits = logits
            self.labels = labels
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if not self.trained:
                print(
                    "TemperatureScaler has not been trained yet. Returning a "
                    "manually tempered input."
                )
            return self._scale(logits)

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale the logits with the optimal temperature.

        Args:
            logits (torch.Tensor): Logits to be scaled.

        Returns:
            torch.Tensor: Scaled logits.
        """
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def fit_predict(
        self, model: nn.Module, calib_loader: DataLoader, progress: bool = True
    ) -> torch.Tensor:
        self.fit(model, calib_loader, save_logits=True, progress=progress)
        calib_logits = self(self.logits)
        return calib_logits
