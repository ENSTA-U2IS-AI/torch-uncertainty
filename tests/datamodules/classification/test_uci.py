from torch_uncertainty.datamodules.classification import HTRU2DataModule


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
