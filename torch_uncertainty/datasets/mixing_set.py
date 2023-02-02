from torchvision import transforms
from torchvision.datasets import ImageFolder


class MixingSet(ImageFolder):
    r"""Adapted from https://github.com/andyzoujm/pixmix/."""

    def __init__(self, root: str, main_dataset: str = "cifar"):
        assert main_dataset in [
            "cifar",
            "imagenet",
        ], """main_dataset should be equal to "cifar" or "imagenet". """

        if main_dataset == "cifar":
            dataset_transform = [
                transforms.Resize(64),
                transforms.RandomCrop(32),
            ]
        elif main_dataset == "resnet":
            dataset_transform = [
                transforms.Resize(256),
                transforms.RandomCrop(224),
            ]

        super().__init__(root, transform=dataset_transform)
