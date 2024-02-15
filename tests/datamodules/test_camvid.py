import pytest

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules.segmentation import CamVidDataModule
from torch_uncertainty.datasets.segmentation import CamVid


class TestCamVidDataModule:
    """Testing the CamVidDataModule datamodule."""

    def test_camvid_main(self):
        # parser = ArgumentParser()
        # parser = CIFAR10DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 16
        dm = CamVidDataModule(root="./data/", batch_size=128)

        assert dm.dataset == CamVid

        dm.dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        with pytest.raises(ValueError):
            dm.setup("xxx")
