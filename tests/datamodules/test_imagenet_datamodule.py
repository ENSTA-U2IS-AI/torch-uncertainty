import pathlib
from argparse import ArgumentParser

import pytest
from torchvision.datasets import ImageNet

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNetDataModule


class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = ImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        args.val_split = 0.1
        dm = ImageNetDataModule(**vars(args))

        assert dm.dataset == ImageNet

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        args.val_split = (
            pathlib.Path(__file__).parent.resolve()
            / "../assets/dummy_indices.yaml"
        )
        dm = ImageNetDataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.setup("fit")
        dm.setup("test")
        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = None
        dm.setup("fit")
        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_ood = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        with pytest.raises(ValueError):
            dm.setup("other")

        for test_alt in ["r", "o", "a"]:
            args.test_alt = test_alt
            dm = ImageNetDataModule(**vars(args))

        with pytest.raises(ValueError):
            dm.setup()

        args.test_alt = "x"
        with pytest.raises(ValueError):
            dm = ImageNetDataModule(**vars(args))

        args.test_alt = None

        for ood_ds in ["inaturalist", "imagenet-o", "textures", "openimage-o"]:
            args.ood_ds = ood_ds
            dm = ImageNetDataModule(**vars(args))
            if ood_ds == "inaturalist":
                dm.eval_ood = True
                dm.dataset = DummyClassificationDataset
                dm.ood_dataset = DummyClassificationDataset
                dm.prepare_data()
                dm.setup("test")
                dm.test_dataloader()

        args.ood_ds = "other"
        with pytest.raises(ValueError):
            dm = ImageNetDataModule(**vars(args))

        args.ood_ds = "svhn"

        for procedure in ["ViT", "A3"]:
            args.procedure = procedure
            dm = ImageNetDataModule(**vars(args))

        args.procedure = "A2"
        with pytest.raises(ValueError):
            dm = ImageNetDataModule(**vars(args))

        args.procedure = None
        args.rand_augment_opt = "rand-m9-n2-mstd0.5"
        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")
