import torch

from torch_uncertainty.models.segmentation.segformer import (
    seg_former_b0,
    seg_former_b1,
    seg_former_b2,
    seg_former_b3,
    seg_former_b4,
    seg_former_b5,
)


class TestSegformer:
    """Testing the Segformer class."""

    def test_main(self):
        seg_former_b1(10)
        seg_former_b2(10)
        seg_former_b3(10)
        seg_former_b4(10)
        seg_former_b5(10)

        model = seg_former_b0(10)
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))
