import torch

from torch_uncertainty.models.classification import (
    batched_inception_time,
    inception_time,
    mimo_inception_time,
    packed_inception_time,
)


class TestInceptionTime:
    """Testing the InceptionTime classes."""

    @torch.no_grad()
    def test_main(self) -> None:
        inception_time(in_channels=1, num_classes=10)(torch.randn(1, 1, 32))

    @torch.no_grad()
    def test_mc_dropout(self) -> None:
        inception_time(in_channels=1, num_classes=10, dropout=0.1).eval()(torch.randn(1, 1, 32))


class TestPackedInceptionTime:
    """Testing the ResNet packed class."""

    @torch.no_grad()
    def test_main(self) -> None:
        packed_inception_time(
            in_channels=1, num_classes=10, num_estimators=2, alpha=2, gamma=1
        )(torch.randn(1, 1, 32))


class TestBatchedInceptionTime:
    """Testing the ResNet batched class."""

    @torch.no_grad()
    def test_main(self) -> None:
        batched_inception_time(in_channels=1, num_classes=10, num_estimators=2)(torch.randn(1, 1, 32))



class TestMIMOInceptionTime:
    """Testing the ResNet MIMO class."""

    @torch.no_grad()
    def test_main(self) -> None:
        mimo_inception_time(in_channels=1, num_classes=10, num_estimators=2).train()(
            torch.rand((2, 1, 28))
        )
