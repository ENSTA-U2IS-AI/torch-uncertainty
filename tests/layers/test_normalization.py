import pytest
import torch

from torch_uncertainty.layers.normalization import (
    FilterResponseNorm2d,
    MCBatchNorm2d,
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


class TestMCBatchNorm2d:
    """Testing the MCBatchNorm2d layer."""

    def test_main(self):
        """Test initialization."""
        bn = MCBatchNorm2d(1, 1)
        bn(torch.randn(1, 1, 20, 20))

    def test_errors(self):
        """Test errors."""
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 0)
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 1.5)
