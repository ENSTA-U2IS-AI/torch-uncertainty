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
        packed_wideresnet28x10(1, 2, 2, 1, 1, 10, style="imagenet")

        with pytest.raises(ValueError):
            _PackedWideResNet(27, 20, 3, 10)


class TestMaskedWide:
    """Testing the WideResNet masked class."""

    def test_main(self):
        masked_wideresnet28x10(1, 2, 2, 1, 10, style="imagenet")

        with pytest.raises(ValueError):
            _MaskedWideResNet(27, 20, 3, 10, 4)


class TestBatchedWide:
    """Testing the WideResNet batched class."""

    def test_main(self):
        batched_wideresnet28x10(1, 2, 1, 10, style="imagenet")

        with pytest.raises(ValueError):
            _BatchWideResNet(27, 20, 3, 10, 4)


class TestMIMOWide:
    """Testing the WideResNet mimo class."""

    def test_main(self):
        model = mimo_wideresnet28x10(1, 10, 2, style="cifar")
        model(torch.rand((2, 1, 28, 28)))

        with pytest.raises(ValueError):
            _MIMOWideResNet(27, 20, 3, 10, 4, 0.0)
