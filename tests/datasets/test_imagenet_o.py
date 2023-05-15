# fmt:off
import pytest

from torch_uncertainty.datasets import ImageNetO


# fmt:on
class TestImageNetO:
    """Testing the ImageNetO dataset class."""

    def test_nodataset(self):
        with pytest.raises(FileNotFoundError):
            _ = ImageNetO("./.data")
