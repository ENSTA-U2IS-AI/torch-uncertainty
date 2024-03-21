import torch

from torch_uncertainty.models.segmentation.segformer import (
    segformer_b0,
    segformer_b1,
    segformer_b2,
    segformer_b3,
    segformer_b4,
    segformer_b5,
)


class TestSegformer:
    """Testing the Segformer class."""

    def test_main(self):
        segformer_b1(10)
        segformer_b2(10)
        segformer_b3(10)
        segformer_b4(10)
        segformer_b5(10)

        model = segformer_b0(10)
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))
