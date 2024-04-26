from torch_uncertainty.models.vgg.packed import packed_vgg
from torch_uncertainty.models.vgg.std import vgg


class TestStdVGG:
    """Testing the VGG std class."""

    def test_main(self):
        vgg(in_channels=1, num_classes=10, arch=11, style="cifar")


class TestPackedVGG:
    """Testing the VGG packed class."""

    def test_main(self):
        packed_vgg(
            in_channels=2,
            num_classes=10,
            arch=11,
            alpha=2,
            num_estimators=2,
            gamma=1,
        )
