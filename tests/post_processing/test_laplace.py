import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.model import dummy_model
from torch_uncertainty.post_processing import LaplaceApprox, PostProcessing


class TestPostProcessing:
    """Testing the PostProcessing class."""

    def test_errors(self):
        PostProcessing.__abstractmethods__ = set()
        pp = PostProcessing(nn.Identity())
        pp.fit(None)
        pp.forward(None)


class TestLaplace:
    """Testing the LaplaceApprox class."""

    def test_training(self):
        dl = DataLoader(TensorDataset(torch.randn(16, 1), torch.randn(16, 10)), batch_size=5)
        la = LaplaceApprox(
            task="classification",
            model=dummy_model(1, 10),
        )
        la.fit(dl)
        la(torch.randn(1, 1))
        la = LaplaceApprox(task="classification")
        la.set_model(dummy_model(1, 10))
