# fmt:off
import pytest

from torch_uncertainty.datasets import ImageNetR


# fmt:on
class TestImageNetR:
    """Testing the ImageNetR dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = ImageNetR("./.data")
