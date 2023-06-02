# fmt:off
import pytest

from torch_uncertainty.datasets import CIFAR10_C, CIFAR100_C


# fmt:on
class TestCIFAR10_C:
    """Testing the CIFAR10_C dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR10_C("./.data")


class TestCIFAR100_C:
    """Testing the CIFAR100_C dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR100_C("./.data")
