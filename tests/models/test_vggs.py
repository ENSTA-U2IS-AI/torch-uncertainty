import torch

from torch_uncertainty.models.vgg.packed import packed_vgg11, packed_vgg16
from torch_uncertainty.models.vgg.std import vgg13, vgg19


class TestStdVGG:
    """Testing the VGG std class."""

    def test_main(self):
        vgg13(1, 10, style="cifar")
        vgg19(1, 10, norm=torch.nn.BatchNorm2d)


class TestPackedVGG:
    """Testing the VGG packed class."""

    def test_main(self):
        packed_vgg11(2, 10, 2, 2, 1)
        packed_vgg16(2, 10, 2, 2, 1)
