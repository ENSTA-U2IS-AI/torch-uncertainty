import pytest

from torch_uncertainty.datasets import MUAD


class TestMUAD:
    """Testing the MUAD dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = MUAD("./.data", split="train")
