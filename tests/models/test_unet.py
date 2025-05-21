import pytest
import torch

from torch_uncertainty.models.segmentation import (
    batched_small_unet,
    batched_unet,
    masked_small_unet,
    masked_unet,
    mimo_small_unet,
    mimo_unet,
    packed_small_unet,
    packed_unet,
    small_unet,
    unet,
)
from torch_uncertainty.models.segmentation.unet.standard import check_unet_parameters


class TestUNet:
    def test_standard_main(self) -> None:
        model = unet(1, 3)
        model.train()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (2, 3, 20, 20)

        model = small_unet(1, 3, bilinear=True)
        model.eval()
        out = model(torch.randn(1, 1, 20, 20))
        assert out.shape == (1, 3, 20, 20)

    def test_packed_main(self) -> None:
        model = packed_unet(1, 3, num_estimators=2, alpha=2)
        model.train()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

        model = packed_small_unet(1, 3, bilinear=True)
        model.eval()
        out = model(torch.randn(1, 1, 20, 20))
        assert out.shape == (1, 3, 20, 20)

    def test_mimo_main(self) -> None:
        model = mimo_unet(1, 3, num_estimators=2)
        model.train()
        out = model(torch.randn(4, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

        model = mimo_small_unet(1, 3, num_estimators=2, bilinear=True)
        model.eval()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

    def test_batched_main(self) -> None:
        model = batched_unet(1, 3, num_estimators=2)
        model.train()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

        model = batched_small_unet(1, 3, num_estimators=2, bilinear=True)
        model.eval()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

    def test_masked_main(self) -> None:
        model = masked_unet(1, 3, num_estimators=2, scale=2)
        model.train()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

        model = masked_small_unet(1, 3, num_estimators=2, scale=2, bilinear=True)
        model.eval()
        out = model(torch.randn(2, 1, 20, 20))
        assert out.shape == (4, 3, 20, 20)

    def test_failures(self) -> None:
        with pytest.raises(ValueError):
            check_unet_parameters(1, 3, [32, 64], bilinear=True)
        with pytest.raises(ValueError):
            check_unet_parameters(1, 3, [32, 64, 128, 256.0, 512], bilinear=True)
        with pytest.raises(ValueError):
            check_unet_parameters(1, 3, [32, 64, -128, 256, 512], bilinear=True)
        with pytest.raises(TypeError):
            check_unet_parameters(1.0, 3, [32, 64, 128, 256, 512], bilinear=True)
        with pytest.raises(TypeError):
            check_unet_parameters(1, 3.0, [32, 64, 128, 256, 512], bilinear=True)
        with pytest.raises(TypeError):
            check_unet_parameters(1, 3, [32, 64, 128, 256, 512], bilinear=1)
        with pytest.raises(ValueError):
            check_unet_parameters(-1, 3, [32, 64, 128, 256, 512], bilinear=True)
        with pytest.raises(ValueError):
            check_unet_parameters(1, -3, [32, 64, 128, 256, 512], bilinear=True)
