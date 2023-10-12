# fmt:off
import pytest

from torch_uncertainty.datasets.classification import ImageNetR


# fmt:on
class TestImageNetR:
    """Testing the ImageNetR dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetR("./.data")
