import pytest

from torch_uncertainty.datasets.segmentation import Cityscapes


class TestCityscapes:
    """Testing the Cityscapes dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = Cityscapes("./.data")
