# fmt:off
from torchvision.transforms import ToTensor

from .dataset import DummyDataset
from .transform import DummyTransform


# fmt:on
class TestDummyDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        dataset = DummyDataset(
            "./.data", transform=ToTensor(), target_transform=DummyTransform()
        )
        _ = len(dataset)
        _, _ = dataset[0]

    def test_dataset_notransform(self):
        dataset = DummyDataset("./.data")
        _ = len(dataset)
        _, _ = dataset[0]
