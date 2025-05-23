import torch

from torch_uncertainty.models.segmentation.segformer import (
    seg_former,
)


class TestSegformer:
    """Testing the Segformer class."""

    @torch.no_grad()
    def test_main(self) -> None:
        model = seg_former(10, 0)
        seg_former(10, 1)
        model(torch.randn(1, 3, 32, 32))
