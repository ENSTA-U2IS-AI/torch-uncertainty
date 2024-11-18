import pytest

from torch_uncertainty.datamodules.classification import (
    BankMarketingDataModule,
    DOTA2GamesDataModule,
    HTRU2DataModule,
    OnlineShoppersDataModule,
    SpamBaseDataModule,
)


class TestHTRU2DataModule:
    """Testing the HTRU2DataModule datamodule class."""

    def test_htru2(self):
        dm = HTRU2DataModule(root="./data/", batch_size=128)

        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.setup("test")
        dm.test_dataloader()

        dm = HTRU2DataModule(root="./data/", batch_size=128, val_split=0.1)

        dm.prepare_data()
        dm.setup()

        with pytest.raises(ValueError):
            dm.setup("other")

        dm = BankMarketingDataModule(root="./data/", batch_size=128)
        dm = DOTA2GamesDataModule(root="./data/", batch_size=128)
        dm = OnlineShoppersDataModule(root="./data/", batch_size=128)
        dm = SpamBaseDataModule(root="./data/", batch_size=128)
