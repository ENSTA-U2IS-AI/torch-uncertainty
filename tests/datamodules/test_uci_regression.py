from functools import partial

from tests._dummies.dataset import DummyRegressionDataset
from torch_uncertainty.datamodules import UCIRegressionDataModule


class TestUCIRegressionDataModule:
    """Testing the UCIRegressionDataModule datamodule class."""

    def test_uci_regression(self):
        dm = UCIRegressionDataModule(
            dataset_name="kin8nm", root="./data/", batch_size=128
        )

        dm.dataset = partial(DummyRegressionDataset, num_samples=64)
        dm.prepare_data()
        dm.val_split = 0.5
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = 0
        dm.setup()
