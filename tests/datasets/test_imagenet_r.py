import pytest

from torch_uncertainty.datasets.classification import ImageNetR


class TestImageNetR:
    """Testing the ImageNetR dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetR("./.data")
