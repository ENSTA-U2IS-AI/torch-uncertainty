import pytest
import torch

from torch_uncertainty.models.classification import wideresnet28x10
from torch_uncertainty.models.classification.wideresnet.batched import (
    _BatchWideResNet,
    batched_wideresnet28x10,
)
from torch_uncertainty.models.classification.wideresnet.masked import (
    _MaskedWideResNet,
    masked_wideresnet28x10,
)
from torch_uncertainty.models.classification.wideresnet.mimo import (
    _MIMOWideResNet,
    mimo_wideresnet28x10,
)
from torch_uncertainty.models.classification.wideresnet.packed import (
    _PackedWideResNet,
    packed_wideresnet28x10,
)


class TestStdWide:
    def test_main(self) -> None:
        wideresnet28x10(in_channels=1, num_classes=10, style="imagenet")

    def test_error(self) -> None:
        with pytest.raises(ValueError):
            wideresnet28x10(in_channels=1, num_classes=10, style="test")


class TestPackedResnet:
    """Testing the WideResNet packed class."""

    def test_main(self) -> None:
        packed_wideresnet28x10(
            in_channels=1,
            num_estimators=2,
            alpha=2,
            groups=1,
            gamma=1,
            num_classes=10,
            style="imagenet",
            conv_bias=False,
        )

        with pytest.raises(ValueError):
            _PackedWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                conv_bias=False,
                dropout_rate=0.0,
            )

        with pytest.raises(ValueError):
            packed_wideresnet28x10(
                in_channels=1,
                num_classes=10,
                num_estimators=2,
                alpha=2,
                groups=1,
                gamma=1,
                style="test",
            )


class TestMaskedWide:
    """Testing the WideResNet masked class."""

    def test_main(self) -> None:
        masked_wideresnet28x10(
            in_channels=1,
            num_classes=10,
            num_estimators=2,
            scale=2.0,
            groups=1,
            style="imagenet",
            conv_bias=False,
        )

        with pytest.raises(ValueError):
            _MaskedWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                conv_bias=False,
                dropout_rate=0.0,
            )

        with pytest.raises(ValueError):
            masked_wideresnet28x10(
                in_channels=1,
                num_classes=10,
                num_estimators=2,
                scale=2.0,
                groups=1,
                style="test",
            )


class TestBatchedWide:
    """Testing the WideResNet batched class."""

    def test_main(self) -> None:
        batched_wideresnet28x10(
            in_channels=1,
            num_classes=10,
            num_estimators=2,
            groups=1,
            style="imagenet",
            conv_bias=False,
        )

        with pytest.raises(ValueError):
            _BatchWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                conv_bias=False,
                dropout_rate=0.0,
            )

        with pytest.raises(ValueError):
            batched_wideresnet28x10(in_channels=1, num_classes=10, num_estimators=2, style="test")


class TestMIMOWide:
    """Testing the WideResNet mimo class."""

    def test_main(self) -> None:
        model = mimo_wideresnet28x10(in_channels=1, num_classes=10, num_estimators=2, style="cifar")
        model(torch.rand((2, 1, 28, 28)))

        with pytest.raises(ValueError):
            _MIMOWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                dropout_rate=0.0,
                conv_bias=False,
            )
        with pytest.raises(ValueError):
            mimo_wideresnet28x10(in_channels=1, num_classes=10, num_estimators=2, style="test")
