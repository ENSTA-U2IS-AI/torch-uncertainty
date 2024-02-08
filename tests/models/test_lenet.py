import pytest
import torch
from torch import nn

from torch_uncertainty.models.lenet import bayesian_lenet, lenet, packed_lenet


class TestLeNet:
    """Testing the LeNet model."""

    def test_main(self):
        """Test initialization."""
        model = lenet(1, 1, norm=nn.BatchNorm2d)
        model.eval()
        model(torch.randn(1, 1, 20, 20))

        model = lenet(1, 1, norm=nn.Identity)
        model.eval()
        model(torch.randn(1, 1, 20, 20))

        packed_lenet(1, 1)
        bayesian_lenet(1, 1)
        bayesian_lenet(1, 1, 1, 1, 1, 0, 1)

    def test_errors(self):
        with pytest.raises(ValueError):
            lenet(1, 1, norm=nn.InstanceNorm2d)
