# fmt:off
import pytest

from torch_uncertainty.datasets import ImageNetA


# fmt:on
class TestImageNetA:
    """Testing the ImageNetA dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetA("./.data")
