from torch_uncertainty.models.vgg.packed import packed_vgg11
from torch_uncertainty.models.vgg.std import vgg11


class TestStdVGG:
    """Testing the VGG std class."""

    def test_main(self):
        vgg11(1, 10, style="cifar")

    def test_mc_dropout(self):
        vgg11(
            in_channels=1,
            num_classes=10,
            style="cifar",
            num_estimators=3,
        )


class TestPackedVGG:
    """Testing the VGG packed class."""

    def test_main(self):
        packed_vgg11(2, 10, 2, 2, 1)
