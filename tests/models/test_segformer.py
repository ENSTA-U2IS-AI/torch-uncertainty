import torch

from torch_uncertainty.models.segmentation.segformer import (
    seg_former,
)


class TestSegformer:
    """Testing the Segformer class."""

    @torch.no_grad()
    def test_main(self):
        model = seg_former(10, 0)
        seg_former(10, 1)
        seg_former(10, 2)
        seg_former(10, 3)
        seg_former(10, 4)
        seg_former(10, 5)
        model(torch.randn(1, 3, 32, 32))
