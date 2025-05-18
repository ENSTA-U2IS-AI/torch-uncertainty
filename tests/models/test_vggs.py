import pytest

from torch_uncertainty.models.classification import packed_vgg, vgg


class TestVGGs:
    """Testing the VGG std class."""

    def test_main(self) -> None:
        vgg(in_channels=1, num_classes=10, arch=11, style="cifar")
        packed_vgg(
            in_channels=2,
            num_classes=10,
            arch=11,
            alpha=2,
            num_estimators=2,
            gamma=1,
        )

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="Unknown VGG arch"):
            vgg(in_channels=1, num_classes=10, arch=12, style="cifar")
        with pytest.raises(ValueError, match="Unknown VGG arch"):
            packed_vgg(
                in_channels=2,
                num_classes=10,
                arch=12,
                alpha=2,
                num_estimators=2,
                gamma=1,
            )
