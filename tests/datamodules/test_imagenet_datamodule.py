# fmt:off
from argparse import ArgumentParser

import pytest
from torchvision.datasets import ImageNet

from torch_uncertainty.datamodules import ImageNetDataModule

from .._dummies.dataset import DummyClassificationDataset



class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = ImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        dm = ImageNetDataModule(**vars(args))

        assert dm.dataset == ImageNet

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.evaluate_ood = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        for test_alt in ["r", "o", "a"]:
            args.test_alt = test_alt
            dm = ImageNetDataModule(**vars(args))

        with pytest.raises(ValueError):
            dm.setup()

        args.test_alt = "x"
        with pytest.raises(ValueError):
            dm = ImageNetDataModule(**vars(args))

        args.test_alt = None

        for ood_ds in ["inaturalist", "imagenet-o", "textures"]:
            args.ood_ds = ood_ds
            dm = ImageNetDataModule(**vars(args))

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
