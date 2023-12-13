import pytest
import torch

from torch_uncertainty.models.wideresnet.batched import (
    _BatchWideResNet,
    batched_wideresnet28x10,
)
from torch_uncertainty.models.wideresnet.masked import (
    _MaskedWideResNet,
    masked_wideresnet28x10,
)
from torch_uncertainty.models.wideresnet.mimo import (
    _MIMOWideResNet,
    mimo_wideresnet28x10,
)
from torch_uncertainty.models.wideresnet.packed import (
    _PackedWideResNet,
    packed_wideresnet28x10,
)
from torch_uncertainty.models.wideresnet.std import _WideResNet, wideresnet28x10


class TestMonteCarloDropoutResnet:
    """Testing the WideResNet MC Dropout."""

    def test_main(self):
        wideresnet28x10(
            in_channels=1,
            num_classes=2,
            groups=1,
            style="imagenet",
            num_estimators=3,
            last_layer_dropout=True,
        )
        wideresnet28x10(
            in_channels=1,
            num_classes=2,
            groups=1,
            style="imagenet",
            num_estimators=3,
            last_layer_dropout=False,
        )

        with pytest.raises(ValueError):
            _WideResNet(27, 20, 3, 10, 0.3)


class TestPackedResnet:
    """Testing the WideResNet packed class."""

    def test_main(self):
        packed_wideresnet28x10(
            in_channels=1,
            num_estimators=2,
            alpha=2,
            groups=1,
            gamma=1,
            num_classes=10,
            style="imagenet",
        )

        with pytest.raises(ValueError):
            _PackedWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                dropout_rate=0.0,
            )


class TestMaskedWide:
    """Testing the WideResNet masked class."""

    def test_main(self):
        masked_wideresnet28x10(
            in_channels=1,
            num_classes=10,
            num_estimators=2,
            scale=2.0,
            groups=1,
            style="imagenet",
        )

        with pytest.raises(ValueError):
            _MaskedWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                dropout_rate=0.0,
            )


class TestBatchedWide:
    """Testing the WideResNet batched class."""

    def test_main(self):
        batched_wideresnet28x10(
            in_channels=1,
            num_classes=10,
            num_estimators=2,
            groups=1,
            style="imagenet",
        )

        with pytest.raises(ValueError):
            _BatchWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                dropout_rate=0.0,
            )


class TestMIMOWide:
    """Testing the WideResNet mimo class."""

    def test_main(self):
        model = mimo_wideresnet28x10(
            in_channels=1, num_classes=10, num_estimators=2, style="cifar"
        )
        model(torch.rand((2, 1, 28, 28)))

        with pytest.raises(ValueError):
            _MIMOWideResNet(
                depth=27,
                widen_factor=20,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                dropout_rate=0.0,
            )
