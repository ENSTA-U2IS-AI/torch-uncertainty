# fmt: off
from typing import Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# fmt: on
class Scaler(nn.Module):
    """
    Virtual class for scaling post-processing for calibrated probabilities.

    Args:
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
        max_iter (int, optional): Maximum number of iterations for the
            optimizer. Defaults to 100.
        device (Optional[Literal["cpu", "cuda"]], optional): Device to use
            for optimization. Defaults to None.

    Reference:
        Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.
    """

    criterion = nn.CrossEntropyLoss()
    trained = False

    def __init__(
        self,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        super().__init__()
        self.device = device

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def fit(
        self,
        model: nn.Module,
        calibration_set: Dataset,
        save_logits: bool = False,
        progress: bool = True,
    ) -> "Scaler":
        """
        Fit the temperature parameters to the calibration data.

        Args:
            model (nn.Module): Model to calibrate.
            calibration_set (Dataset): Calibration dataset.
            save_logits (bool, optional): Whether to save the logits and
                labels. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.

        Returns:
            TemperatureScaler: Calibrated scaler.
        """
        logits_list = []
        labels_list = []
        calibration_dl = DataLoader(
            calibration_set, batch_size=32, shuffle=False, drop_last=False
        )
        with torch.no_grad():
            for input, label in tqdm(calibration_dl, disable=not progress):
                input = input.to(self.device)
                logits = model(input)
                logits_list.append(logits)
                labels_list.append(label)
        logits = torch.cat(logits_list).detach().to(self.device)
        labels = torch.cat(labels_list).detach().to(self.device)

        optimizer = optim.LBFGS(
            self.temperature, lr=self.lr, max_iter=self.max_iter
        )

        def calib_eval() -> float:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(calib_eval)
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
        raise NotImplementedError()

    def fit_predict(
        self,
        model: nn.Module,
        calibration_set: Dataset,
        progress: bool = True,
    ) -> torch.Tensor:
        self.fit(model, calibration_set, save_logits=True, progress=progress)
        calib_logits = self(self.logits)
        return calib_logits

    @property
    def temperature(self) -> list:
        raise NotImplementedError()
