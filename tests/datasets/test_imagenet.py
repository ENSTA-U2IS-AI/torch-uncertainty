import pytest

from torch_uncertainty.datasets.classification import (
    ImageNetA,
    ImageNetO,
    ImageNetR,
)


class TestImageNetA:
    """Testing the ImageNetA dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetA("./.data")


class TestImageNetO:
    """Testing the ImageNetO dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetO("./.data")


class TestImageNetR:
    """Testing the ImageNetR dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetR("./.data")
