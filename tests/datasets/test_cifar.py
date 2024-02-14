import pytest

from torch_uncertainty.datasets.classification import (
    CIFAR10C,
    CIFAR10H,
    CIFAR100C,
)


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


class TestCIFAR10H:
    """Testing the CIFAR10H dataset class."""

    def test_nodataset_nodownload(self):
        with pytest.raises(RuntimeError):
            _ = CIFAR10H("./.data", download=False)
