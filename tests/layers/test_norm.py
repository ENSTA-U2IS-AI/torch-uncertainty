import pytest
import torch

from torch_uncertainty.layers.channel_layer_norm import ChannelLayerNorm
from torch_uncertainty.layers.filter_response_norm import (
    FilterResponseNorm1d,
    FilterResponseNorm2d,
    FilterResponseNorm3d,
    _FilterResponseNormNd,
)
from torch_uncertainty.layers.mc_batch_norm import (
    MCBatchNorm1d,
    MCBatchNorm2d,
    MCBatchNorm3d,
)


class TestFilterResponseNorm:
    """Testing the FRN2d layer."""

    def test_main(self):
        """Test initialization."""
        frn = FilterResponseNorm2d(1)
        frn(torch.randn(1, 1, 20, 20))
        FilterResponseNorm1d(1)
        FilterResponseNorm3d(1)

    def test_errors(self):
        """Test errors."""
        with pytest.raises(ValueError):
            _FilterResponseNormNd(-1, 1)
        with pytest.raises(ValueError):
            FilterResponseNorm2d(0)
        with pytest.raises(ValueError):
            FilterResponseNorm2d(1.5)


class TestMCBatchNorm:
    """Testing the MCBatchNorm layers."""

    def test_main(self):
        """Test initialization."""
        bn = MCBatchNorm1d(1, 1)
        bn(torch.randn(1, 1, 2))
        bn = MCBatchNorm2d(1, 1)
        bn(torch.randn(1, 1, 1, 2))
        bn = MCBatchNorm3d(1, 1)
        bn(torch.randn(1, 1, 1, 1, 2))

    def test_errors(self):
        """Test errors."""
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 0)
        with pytest.raises(ValueError):
            MCBatchNorm2d(1, 1.5)

        layer1d = MCBatchNorm1d(1, 1)
        layer2d = MCBatchNorm2d(1, 1)
        layer3d = MCBatchNorm3d(1, 1)
        with pytest.raises(ValueError):
            layer1d(torch.randn(1, 1, 1, 20))
        with pytest.raises(ValueError):
            layer2d(torch.randn(1, 1, 1, 1, 20))
        with pytest.raises(ValueError):
            layer3d(torch.randn(1, 1, 1, 1, 1, 20))


class TestChannelLayerNorm:
    """Testing the FRN2d layer."""

    def test_main(self):
        """Test initialization."""
        cln = ChannelLayerNorm(1)
        cln(torch.randn(1, 1, 4, 4))
        cln = ChannelLayerNorm(18)
        cln(torch.randn(1, 18, 2, 3))
