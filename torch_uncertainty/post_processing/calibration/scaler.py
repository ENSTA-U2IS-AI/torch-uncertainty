import logging
from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader
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
        dataloader: DataLoader,
        save_logits: bool = False,
        progress: bool = True,
    ) -> None:
        """Fit the temperature parameters to the calibration data.

        Args:
            dataloader (DataLoader): Dataloader with the calibration data.
            save_logits (bool, optional): Whether to save the logits and
                labels in memory. Defaults to False.
            progress (bool, optional): Whether to show a progress bar.
                Defaults to True.
        """
        if self.model is None or isinstance(self.model, nn.Identity):
            logging.warning(
                "model is None. Fitting the temperature scaling on the x of the dataloader."
            )
            self.model = nn.Identity()

        all_logits = []
        all_labels = []
        calibration_dl = dataloader
        with torch.no_grad():
            for inputs, labels in tqdm(calibration_dl, disable=not progress):
                logits = self.model(inputs.to(self.device))
                all_logits.append(logits)
                all_labels.append(labels)
            all_logits = torch.cat(all_logits).to(self.device)
            all_labels = torch.cat(all_labels).to(self.device)

        if all_logits.dim() == 2 and all_logits.shape[1] == 1:
            all_logits = all_logits.squeeze(1)
        if all_logits.dim() == 1:
            confidence = torch.log(all_logits.sigmoid())
            all_labels = all_labels.to(dtype=torch.float32)
            all_logits = torch.stack([torch.log(1 - confidence), torch.log(confidence)], dim=1)

        optimizer = LBFGS(self.temperature, lr=self.lr, max_iter=self.max_iter)

        def calib_eval() -> float:
            optimizer.zero_grad()
            loss = self.criterion(self._scale(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(calib_eval)
        self.trained = True
        if save_logits:
            self.logits = all_logits
            self.labels = all_labels

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        if self.model is None or not self.trained:
            logging.error(
                "TemperatureScaler has not been trained yet. Returning manually tempered inputs."
            )
        return self._scale(self.model(inputs))

    @abstractmethod
    def _scale(self, logits: Tensor) -> Tensor:
        """Scale the logits with the optimal temperature.

        Args:
            logits (Tensor): Logits to be scaled.

        Returns:
            Tensor: Scaled logits.
        """
        ...

    def fit_predict(
        self,
        dataloader: DataLoader,
        progress: bool = True,
    ) -> Tensor:
        self.fit(dataloader, save_logits=True, progress=progress)
        return self(self.logits)

    @property
    @abstractmethod
    def temperature(self) -> list: ...
