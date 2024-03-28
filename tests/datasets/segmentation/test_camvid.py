import pytest

from torch_uncertainty.datasets.segmentation import CamVid


class TestCamVid:
    """Testing the CamVid dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = CamVid("./.data")
