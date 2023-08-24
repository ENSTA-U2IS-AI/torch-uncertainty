from torch_uncertainty.models.wideresnet.batched import batched_wideresnet28x10
from torch_uncertainty.models.wideresnet.masked import masked_wideresnet28x10
from torch_uncertainty.models.wideresnet.packed import packed_wideresnet28x10


class TestPackedResnet:
    """Testing the WideResNet packed class."""

    def test_main(self):
        packed_wideresnet28x10(1, 2, 2, 1, 1, 10, style="cifar")


class TestMaskedWide:
    """Testing the WideResNet masked class."""

    def test_main(self):
        masked_wideresnet28x10(1, 2, 2, 1, 10, style="cifar")


class TestBatchedWide:
    """Testing the WideResNet batched class."""

    def test_main(self):
        batched_wideresnet28x10(1, 2, 1, 10, style="cifar")
