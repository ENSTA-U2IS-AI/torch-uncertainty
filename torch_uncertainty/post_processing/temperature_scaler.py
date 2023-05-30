# fmt: off
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


# fmt: on
class TemperatureScaler(nn.Module):
    """
    Temperature scaling post-processing for calibrated probabilities.

    Args:
        init_value (float, optional): Initial value for the temperature.
            Defaults to 1.

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
        lr: float = 0.01,
        max_iter: int = 50,
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        if init_val <= 0:
            raise ValueError("Initial temperature value must be positive.")

        self.temperature = nn.Parameter(torch.ones(1) * init_val).to(device)
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

        self.temperature = nn.Parameter(torch.ones(1) * val)

    def fit(
        self, model: nn.Module, val_loader: DataLoader
    ) -> "TemperatureScaler":
        """
        Fit the temperature to the validation data.

        Args:
            model (nn.Module): Model to calibrate.
            val_loader (DataLoader): Validation dataloader.

        Returns:
            TemperatureScaler: Calibrated scaler.
        """
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(self.device)
                logits = model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        optimizer = optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )

        def eval() -> torch.Tensor:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        self.trained = True

        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            print(
                "TemperatureScaler has not been trained yet. Returning a "
                "manually tempered input."
            )
        return self._scale(logits)

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale the logits by the temperature.

        Args:
            logits (torch.Tensor): Logits to scale.

        Returns:
            torch.Tensor: Scaled logits.
        """
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature
