# fmt: off
from torchvision import transforms
from torchvision.datasets import ImageFolder


# fmt: on
class MixingSet(ImageFolder):
    r"""Datasets used by PixMix augmentations.

    Args:
        root (str): Root directory of dataset.
        main_dataset (str): Main dataset to be mixed with.
            Should be 'cifar' or 'imagenet'.

    Reference:
        Adapted from https://github.com/andyzoujm/pixmix/.
    """

    def __init__(self, root: str, main_dataset: str = "cifar"):
        if main_dataset == "cifar":
            dataset_transform = [
                transforms.Resize(64),
                transforms.RandomCrop(32),
            ]
        elif main_dataset == "imagenet":
            dataset_transform = [
                transforms.Resize(256),
                transforms.RandomCrop(224),
            ]
        else:
            raise ValueError("main_dataset should be 'cifar' or 'imagenet'.")

        super().__init__(root, transform=dataset_transform)
