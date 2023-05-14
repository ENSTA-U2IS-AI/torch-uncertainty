# fmt:off
from .dummy_dataset import DummyDataset


# fmt:on
class TestDummyDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        dataset = DummyDataset("./.data")
        _ = len(dataset)
        _, _ = dataset[0]
