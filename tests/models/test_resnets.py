import torch

from torch_uncertainty.models.resnet.batched import (
    batched_resnet34,
    batched_resnet101,
    batched_resnet152,
)
from torch_uncertainty.models.resnet.masked import (
    masked_resnet34,
    masked_resnet101,
)
from torch_uncertainty.models.resnet.mimo import (
    mimo_resnet34,
    mimo_resnet101,
    mimo_resnet152,
)
from torch_uncertainty.models.resnet.packed import (
    packed_resnet34,
    packed_resnet101,
    packed_resnet152,
)
from torch_uncertainty.models.resnet.std import (
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


class TestStdResnet:
    """Testing the ResNet std class."""

    def test_main(self):
        resnet34(1, 10, 1)
        resnet101(1, 10, 1)
        resnet152(1, 10, 1)

        model = resnet50(1, 10, 1)
        with torch.no_grad():
            model(torch.randn(2, 1, 32, 32))
            model.feats_forward(torch.randn(2, 1, 32, 32))

    def test_mc_dropout(self):
        resnet34(1, 10, 1, num_estimators=5)
        resnet101(1, 10, 1, num_estimators=5)
        resnet152(1, 10, 1, num_estimators=5)

        model = resnet50(1, 10, 1, num_estimators=5)
        with torch.no_grad():
            model(torch.randn(2, 1, 32, 32))


class TestPackedResnet:
    """Testing the ResNet packed class."""

    def test_main(self):
        packed_resnet34(1, 2, 2, 1, 10, 1)
        packed_resnet101(1, 2, 2, 1, 10, 1)
        model = packed_resnet152(1, 2, 2, 1, 10, 1)

        out = model.check_config(
            {"alpha": 2, "gamma": 1, "groups": 1, "num_estimators": 2}
        )
        assert out
        out = model.check_config(
            {"alpha": 1, "gamma": 1, "groups": 1, "num_estimators": 2}
        )
        assert not out


class TestMaskedResnet:
    """Testing the ResNet masked class."""

    def test_main(self):
        masked_resnet34(1, 2, 2, 1, 10)
        masked_resnet101(1, 2, 2, 1, 10)


class TestBatchedResnet:
    """Testing the ResNet batched class."""

    def test_main(self):
        batched_resnet34(1, 2, 1, 10)
        batched_resnet101(1, 2, 1, 10)
        batched_resnet152(1, 2, 1, 10)


class TestMIMOResnet:
    """Testing the ResNet MIMO class."""

    def test_main(self):
        model = mimo_resnet34(1, 10, 2, style="cifar")
        model.train()
        model(torch.rand((2, 1, 28, 28)))
        mimo_resnet101(1, 10, 2)
        mimo_resnet152(1, 10, 2)
