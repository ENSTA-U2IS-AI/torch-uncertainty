from abc import abstractmethod

from torch import Tensor

from torch_uncertainty.post_processing.abstract import PostProcessing


class Conformal(PostProcessing):
    @abstractmethod
    def conformal(self, inputs: Tensor) -> tuple[Tensor, Tensor]: ...

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return self.conformal(inputs)
