# fmt:off
from torchvision.transforms import ToTensor

from ..transforms.dummy_transform import DummyTransform
from .dummy_dataset import DummyDataset


# fmt:on
class TestDummyDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        dataset = DummyDataset(
            "./.data", transform=ToTensor(), target_transform=DummyTransform()
        )
        _ = len(dataset)
        _, _ = dataset[0]
