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


@StochasticModel
class DummyModelMix(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesConv2d(1, 10, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv2d(10, 1, 1)

    def forward(self, x):
        y = self.relu(self.layer(x))
        y = self.layer2(y)
        return y


class TestStochasticModel:
    """Testing the StochasticModel decorator."""

    def test_main(self):
        model = DummyModelLinear()
        model.freeze()
        assert model.layer.frozen
        model.unfreeze()
        assert not model.layer.frozen

        model = DummyModelConv()
        model.freeze()
        assert model.layer.frozen
        model.unfreeze()
        assert not model.layer.frozen

    def test_mix(self):
        model = DummyModelMix()
        model.freeze()
        assert model.layer.frozen
        model.unfreeze()
        assert not model.layer.frozen

        state = model.sample()[0]
        keys = state.keys()
        assert list(keys) == [
            "layer.weight",
            "layer.bias",
            "layer2.weight",
            "layer2.bias",
        ]
