import pytest

from torch_uncertainty.datasets.classification import ImageNetA


class TestImageNetA:
    """Testing the ImageNetA dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetA("./.data")
