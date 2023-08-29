import torch

from torch_uncertainty.layers import BayesConv2d, BayesLinear
from torch_uncertainty.models.utils import StochasticModel


@StochasticModel
class DummyModelLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesLinear(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


@StochasticModel
class DummyModelConv(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesConv2d(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


class TestStochasticModel:
    """Testing the ResNet std class."""

    def test_main_linear(self):
        model = DummyModelLinear()
        model.freeze()
        model.unfreeze()

    def test_main_conv(self):
        model = DummyModelConv()
        model.freeze()
        model.unfreeze()
