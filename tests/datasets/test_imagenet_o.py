# fmt:off
import pytest

from torch_uncertainty.datasets.classification import ImageNetO


# fmt:on
class TestImageNetO:
    """Testing the ImageNetO dataset class."""

    def test_nodataset(self):
        with pytest.raises(RuntimeError):
            _ = ImageNetO("./.data")
