import pytest
import torch

from torch_uncertainty.models.segmentation.deeplab import _DeepLabV3, deep_lab_v3_resnet


class TestDeeplab:
    """Testing the Deeplab class."""

    @torch.no_grad()
    def test_main(self):
        model = deep_lab_v3_resnet(10, 50, "v3", 16, True, False).eval()
        model(torch.randn(1, 3, 32, 32))
        model = deep_lab_v3_resnet(10, 50, "v3", 16, False, False).eval()
        model = deep_lab_v3_resnet(10, 101, "v3+", 8, True, False).eval()
        model(torch.randn(1, 3, 32, 32))
        model = deep_lab_v3_resnet(10, 101, "v3+", 8, False, False).eval()

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="Unknown backbone:"):
            _DeepLabV3(10, "other", "v3", 16, True, False)
        with pytest.raises(ValueError, match="output_stride: "):
            deep_lab_v3_resnet(10, 50, "v3", 15, True, False)
        with pytest.raises(ValueError, match="Unknown style: "):
            deep_lab_v3_resnet(10, 50, "v2", 16, True, False)
