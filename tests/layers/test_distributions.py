import pytest

from torch_uncertainty.layers.distributions import (
    AbstractDist,
    LaplaceLayer,
    NormalLayer,
)


class TestDistributions:
    def test(self):
        AbstractDist.__abstractmethods__ = set()
        dist = AbstractDist(dim=1)
        dist.forward(None)

    def test_errors(self):
        with pytest.raises(ValueError):
            NormalLayer(-1, 1)
        with pytest.raises(ValueError):
            NormalLayer(1, -1)
        with pytest.raises(ValueError):
            LaplaceLayer(1, -1)
