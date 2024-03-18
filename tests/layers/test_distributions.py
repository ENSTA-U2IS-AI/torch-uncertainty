import pytest

from torch_uncertainty.layers.distributions import (
    IndptLaplaceLayer,
    IndptNormalLayer,
)


class TestDistributions:
    def test_errors(self):
        with pytest.raises(ValueError):
            IndptNormalLayer(-1, 1)
        with pytest.raises(ValueError):
            IndptNormalLayer(1, -1)
        with pytest.raises(ValueError):
            IndptLaplaceLayer(1, -1)
