# fmt:off
import pytest

from torch_uncertainty.datasets import CIFAR10_H


# fmt:on
class TestCIFAR10_H:
    """Testing the CIFAR10_H dataset class."""

    def test_nodataset_nodownload(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR10_H("./.data", download=False)
