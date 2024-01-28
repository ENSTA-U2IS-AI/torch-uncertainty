import pytest
from torch import nn

from torch_uncertainty.models.lenet import lenet


class TestLeNet:
    """Testing the LeNet model."""

    def test_errors(self):
        with pytest.raises(ValueError):
            lenet(1, 1, norm=nn.InstanceNorm2d)
