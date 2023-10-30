# fmt:off
import pytest

from torch_uncertainty.datasets.classification import CIFAR10C, CIFAR100C



class TestCIFAR10C:
    """Testing the CIFAR10C dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR10C("./.data")


class TestCIFAR100C:
    """Testing the CIFAR100C dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR100C("./.data")
