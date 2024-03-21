import pytest

from torch_uncertainty.layers.distributions import (
    LaplaceLayer,
    NormalLayer,
)


class TestDistributions:
    def test_errors(self):
        with pytest.raises(ValueError):
            NormalLayer(-1, 1)
        with pytest.raises(ValueError):
            NormalLayer(1, -1)
        with pytest.raises(ValueError):
            LaplaceLayer(1, -1)
