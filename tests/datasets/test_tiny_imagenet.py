import pytest

from torch_uncertainty.datasets.classification import TinyImageNet


class TestTinyImageNet:
    """Testing the TinyImageNet dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = TinyImageNet("./.data")
