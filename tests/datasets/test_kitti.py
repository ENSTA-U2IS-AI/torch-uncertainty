import pytest

from torch_uncertainty.datasets import KITTIDepth


class TestKITTIDepth:
    """Testing the KITTIDepth dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = KITTIDepth("./.data", split="train")
