# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import UCIDataModule

from .._dummies.dataset import DummyRegressionDataset


# fmt:on
class TestUCIDataModule:
    """Testing the UCIDataModule datamodule class."""

    def test_UCIRegression(self):
        parser = ArgumentParser()
        parser = UCIDataModule.add_argparse_args(parser)

        args = parser.parse_args("")

        dm = UCIDataModule(dataset_name="kin8nm", **vars(args))

        dm.dataset = DummyRegressionDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = 0
        dm.setup()
