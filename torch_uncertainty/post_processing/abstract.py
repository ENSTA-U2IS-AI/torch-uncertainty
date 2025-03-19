from abc import ABC, abstractmethod

from torch import Tensor, nn
from torch.utils.data import DataLoader


class PostProcessing(ABC, nn.Module):
    def __init__(self, model: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.trained = False

    def set_model(self, model: nn.Module) -> None:
        self.model = model

    @abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        pass
