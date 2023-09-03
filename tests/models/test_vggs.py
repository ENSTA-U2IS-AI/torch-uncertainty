from torch_uncertainty.models.vgg.packed import packed_vgg11, packed_vgg16
from torch_uncertainty.models.vgg.std import vgg13


class TestStdVGG:
    """Testing the VGG std class."""

    def test_main(self):
        vgg13(1, 10, style="cifar")


class TestPackedVGG:
    """Testing the VGG packed class."""

    def test_main(self):
        packed_vgg11(2, 10, 2, 2, 1)
        packed_vgg16(2, 10, 2, 2, 1)
