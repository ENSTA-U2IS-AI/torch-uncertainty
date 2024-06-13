from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import Dataset


class PostProcessing(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = False

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        pass
