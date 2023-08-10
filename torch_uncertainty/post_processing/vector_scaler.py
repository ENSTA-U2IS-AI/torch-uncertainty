# fmt: off
from typing import Literal, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# fmt: on
class VectorScaler(nn.Module):
    """
    Vector scaling post-processing for calibrated probabilities.

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

    criterion = nn.CrossEntropyLoss()
    trained = False

    def __init__(
        self,
        num_classes: int,
        init_w: float = 1,
        init_b: float = 0,
        lr: float = 0.1,
        max_iter: int = 200,
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        super().__init__()
        self.device = device

        if not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer.")
        if num_classes <= 0:
            raise ValueError("The number of classes must be positive.")
        self.num_classes = num_classes

        self.temp_w = nn.Parameter(
            torch.ones(num_classes, device=device) * init_w,
            requires_grad=True,
        )
        self.temp_b = nn.Parameter(
            torch.ones(num_classes, device=device) * init_b,
            requires_grad=True,
        )

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def set_temperature(self, val_w: float, val_b: float) -> None:
        """
        Set the temperature to a fixed value.

        Args:
            val_w (float): Weight temperature value.
            val_b (float): Bias temperature value.
        """
        self.temp_w = nn.Parameter(
            torch.ones(self.num_classes, device=self.device) * val_w,
            requires_grad=True,
        )
        self.temp_b = nn.Parameter(
            torch.ones(self.num_classes, device=self.device) * val_b,
            requires_grad=True,
        )

    def fit(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        save_logits: bool = False,
        progress: bool = True,
    ) -> "VectorScaler":
        """
        Fit the temperature vectors to the calibration data.

        Args:
            model (nn.Module): Model to calibrate.
            calib_loader (DataLoader): Calibration dataloader.
            save_logits (bool, optional): Whether to save the logits and
                labels. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.

        Returns:
            VectorScaler: Calibrated scaler.
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
            [self.temp_w, self.temp_b], lr=self.lr, max_iter=self.max_iter
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
                    "VectorScaler has not been trained yet. Returning a "
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
        return self.temp_w * logits + self.temp_b

    def fit_predict(
        self, model: nn.Module, calib_loader: DataLoader, progress: bool = True
    ) -> torch.Tensor:
        self.fit(model, calib_loader, save_logits=True, progress=progress)
        calib_logits = self(self.logits)
        return calib_logits
