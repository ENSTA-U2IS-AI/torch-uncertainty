import torch
from torch import nn

from torch_uncertainty.layers import BayesConv2d, BayesLinear
from torch_uncertainty.models import StochasticModel


class DummyModelLinear(nn.Module):
    """Dummy model for testing purposes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesLinear(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


class DummyModelConv(nn.Module):
    """Dummy conv model for testing purposes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesConv2d(1, 10, 1)

    def forward(self, x):
        return self.layer(x)


class DummyModelMix(nn.Module):
    """Dummy mix model for testing purposes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BayesConv2d(1, 10, 1, bias=False)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv2d(10, 1, 1)

    def forward(self, x):
        y = self.relu(self.layer(x))
        return self.layer2(y)


class TestStochasticModel:
    """Testing the StochasticModel decorator."""

    def test_main(self):
        model = StochasticModel(DummyModelLinear(), 2)
        model.freeze()
        model(torch.randn(1, 1))
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen
        model.eval()
        model(torch.randn(1, 1))

        model = StochasticModel(DummyModelConv(), 2)
        model.freeze()
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen

    def test_mix(self):
        model = StochasticModel(DummyModelMix(), 2)
        model.freeze()
        assert model.core_model.layer.frozen
        model.unfreeze()
        assert not model.core_model.layer.frozen

        state = model.sample()[0]
        keys = state.keys()
        print(list(keys))
        assert list(keys) == [
            "layer.weight",
            "layer2.weight",
            "layer2.bias",
        ]
