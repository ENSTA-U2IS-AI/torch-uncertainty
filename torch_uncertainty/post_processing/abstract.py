from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import Dataset


class PostProcessing(ABC, nn.Module):
    def __init__(self, model: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.trained = False

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        pass
