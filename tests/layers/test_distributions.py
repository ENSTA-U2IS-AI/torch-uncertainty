import pytest

from torch_uncertainty.layers.distributions import (
    LaplaceLayer,
    NormalLayer,
    TUDist,
)


class TestDistributions:
    def test(self):
        TUDist.__abstractmethods__ = set()
        dist = TUDist(dim=1)
        dist.forward(None)

    def test_errors(self):
        with pytest.raises(ValueError):
            NormalLayer(-1, 1)
        with pytest.raises(ValueError):
            NormalLayer(1, -1)
        with pytest.raises(ValueError):
            LaplaceLayer(1, -1)
