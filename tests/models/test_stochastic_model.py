import torch

from torch_uncertainty.layers import BayesLinear
from torch_uncertainty.models.utils import StochasticModel


@StochasticModel
class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesLinear(1, 10, 1)


class TestStochasticModel:
    """Testing the ResNet std class."""

    def test_main(self):
        model = DummyModel()
        model.freeze()
        model.unfreeze()
