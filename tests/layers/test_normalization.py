import pytest
import torch

from torch_uncertainty.layers.mc_batch_norm import (
    MCBatchNorm1d,
    MCBatchNorm2d,
    MCBatchNorm3d,
)
from torch_uncertainty.layers.normalization import (
    FilterResponseNorm2d,
)


class TestFRN2d:
    """Testing the FRN2d layer."""

    def test_main(self):
        """Test initialization."""
        frn = FilterResponseNorm2d(1)
        frn(torch.randn(1, 1, 20, 20))

    def test_errors(self):
        """Test errors."""
        with pytest.raises(ValueError):
            FilterResponseNorm2d(0)
        with pytest.raises(ValueError):
            FilterResponseNorm2d(1.5)


class TestMCBatchNorm:
    """Testing the MCBatchNorm layers."""

    def test_main(self):
        """Test initialization."""
        bn = MCBatchNorm1d(1, 1)
        bn = MCBatchNorm3d(1, 1)
        bn = MCBatchNorm2d(1, 1)
        bn(torch.randn(1, 1, 20, 20))

    def test_errors(self):
        """Test errors."""
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 0)
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 1.5)
