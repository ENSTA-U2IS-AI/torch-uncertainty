import logging
from typing import Literal

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_uncertainty.post_processing import PostProcessing


class Scaler(PostProcessing):
    criterion = nn.CrossEntropyLoss()
    trained = False

    def __init__(
        self,
        model: nn.Module | None = None,
        lr: float = 0.1,
        max_iter: int = 100,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Virtual class for scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the
                optimizer. Defaults to 100.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use
                for optimization. Defaults to None.

        Reference:
            Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. On calibration
            of modern neural networks. In ICML 2017.
        """
        super().__init__(model)
        self.device = device

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        self.lr = lr

        if max_iter <= 0:
            raise ValueError("Max iterations must be positive.")
        self.max_iter = int(max_iter)

    def fit(
        self,
        calibration_set: Dataset,
        save_logits: bool = False,
        progress: bool = True,
    ) -> None:
        """Fit the temperature parameters to the calibration data.

        Args:
            calibration_set (Dataset): Calibration dataset.
            save_logits (bool, optional): Whether to save the logits and
                labels. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
        """
        logits_list = []
        labels_list = []
        calibration_dl = DataLoader(
            calibration_set, batch_size=32, shuffle=False, drop_last=False
        )
        with torch.no_grad():
            for inputs, labels in tqdm(calibration_dl, disable=not progress):
                logits = self.model(inputs.to(self.device))
                logits_list.append(logits)
                labels_list.append(labels)
        all_logits = torch.cat(logits_list).detach().to(self.device)
        all_labels = torch.cat(labels_list).detach().to(self.device)

        optimizer = optim.LBFGS(
            self.temperature, lr=self.lr, max_iter=self.max_iter
        )

        def calib_eval() -> float:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(calib_eval)
        self.trained = True
        if save_logits:
            self.logits = logits
            self.labels = labels

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if not self.trained:
            logging.error(
                "TemperatureScaler has not been trained yet. Returning "
                "manually tempered inputs."
            )
        return self._scale(self.model(inputs))

    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the logits with the optimal temperature.

        Args:
            logits (Tensor): Logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        raise NotImplementedError

    def fit_predict(
        self,
        calibration_set: Dataset,
        progress: bool = True,
    ) -> Tensor:
        self.fit(calibration_set, save_logits=True, progress=progress)
        return self(self.logits)

    @property
    def temperature(self) -> list:
        raise NotImplementedError
