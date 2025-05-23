import pytest
import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    VGGBaseline,
    WideResNetBaseline,
)
from torch_uncertainty.baselines.regression import MLPBaseline
from torch_uncertainty.baselines.segmentation import (
    DeepLabBaseline,
    SegFormerBaseline,
)


class TestStandardBaseline:
    """Testing the ResNetBaseline baseline class."""

    def test_standard(self) -> None:
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="std",
            arch=18,
            style="cifar",
            groups=1,
        )
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="test",
                arch=18,
                style="cifar",
                groups=1,
            )


class TestStandardWideBaseline:
    """Testing the WideResNetBaseline baseline class."""

    def test_standard(self) -> None:
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="std",
            style="cifar",
            groups=1,
        )
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            WideResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="test",
                style="cifar",
                groups=1,
            )


class TestStandardVGGBaseline:
    """Testing the VGGBaseline baseline class."""

    def test_standard(self) -> None:
        net = VGGBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="std",
            arch=11,
            groups=1,
        )
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            VGGBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="test",
                arch=11,
                groups=1,
            )


class TestStandardMLPBaseline:
    """Testing the MLP baseline class."""

    def test_standard(self) -> None:
        net = MLPBaseline(
            in_features=3,
            output_dim=10,
            loss=nn.MSELoss(),
            version="std",
            hidden_dims=[1],
        )
        _ = net(torch.rand(1, 3))
        for dist_family in ["normal", "laplace", "nig"]:
            MLPBaseline(
                in_features=3,
                output_dim=10,
                loss=nn.MSELoss(),
                version="std",
                hidden_dims=[1],
                dist_family=dist_family,
            )

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            MLPBaseline(
                in_features=3,
                output_dim=10,
                loss=nn.MSELoss(),
                version="test",
                hidden_dims=[1],
            )


class TestStandardSegFormerBaseline:
    """Testing the SegFormer baseline class."""

    def test_standard(self) -> None:
        net = SegFormerBaseline(
            num_classes=10,
            loss=nn.CrossEntropyLoss(),
            version="std",
            arch=0,
        )
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self) -> None:
        with pytest.raises(ValueError):
            SegFormerBaseline(
                num_classes=10,
                loss=nn.CrossEntropyLoss(),
                version="test",
                arch=0,
            )


class TestStandardDeepLabBaseline:
    """Testing the DeepLab baseline class."""

    def test_standard(self) -> None:
        net = DeepLabBaseline(
            num_classes=10,
            loss=nn.CrossEntropyLoss(),
            version="std",
            style="v3",
            output_stride=16,
            arch=50,
            separable=True,
        ).eval()
        _ = net(torch.rand(1, 3, 32, 32))
