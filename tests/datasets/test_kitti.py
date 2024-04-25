import pytest

from torch_uncertainty.datasets import KITTIDepth


class TestMUAD:
    """Testing the MUAD dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = KITTIDepth("./.data", split="train")
