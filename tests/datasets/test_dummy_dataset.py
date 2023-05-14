# fmt:off
from .dummy_dataset import DummyDataset


# fmt:on
class TestDummyDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        _ = DummyDataset("./.data")
