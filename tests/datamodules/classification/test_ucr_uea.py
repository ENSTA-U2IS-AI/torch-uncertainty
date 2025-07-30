from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import UCRUEADataModule


class TestUCRUEADataModule:
    """Testing the UCRUEADataModule datamodule class."""

    def test_ucr_uea_main(self) -> None:
        dm = UCRUEADataModule(
            dataset_name="test",
            batch_size=128,
        )
        dm.dataset = DummyClassificationDataset
        dm.setup()
        dm.setup("test")
