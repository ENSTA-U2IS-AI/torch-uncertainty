import torch

from torch_uncertainty.models.depth import bts_resnet


class TestBTS:
    """Testing the BTS model class."""

    @torch.no_grad()
    def test_main(self):
        model = bts_resnet(50, 1).eval()
        model(torch.randn(1, 3, 32, 32))
        model = bts_resnet(50, 1, dist_family="normal").eval()
        model(torch.randn(1, 3, 32, 32))
