from torchvision.transforms import ToTensor

from .dataset import DummyClassificationDataset, DummyRegressionDataset
from .transform import DummyTransform


class TestDummyClassificationDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        dataset = DummyClassificationDataset(
            "./.data", transform=ToTensor(), target_transform=DummyTransform()
        )
        _ = len(dataset)
        _, _ = dataset[0]

    def test_dataset_notransform(self):
        dataset = DummyClassificationDataset("./.data")
        _ = len(dataset)
        _, _ = dataset[0]


class TestDummyRegressionDataset:
    """Testing the Dummy dataset class."""

    def test_dataset(self):
        dataset = DummyRegressionDataset(
            "./.data",
            transform=DummyTransform(),
            target_transform=DummyTransform(),
        )
        _ = len(dataset)
        _, _ = dataset[0]

    def test_dataset_notransform(self):
        dataset = DummyRegressionDataset("./.data")
        _ = len(dataset)
        _, _ = dataset[0]
