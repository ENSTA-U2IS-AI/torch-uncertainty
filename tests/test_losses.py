# fmt: off
import pytest
import torch
from torch import nn

from torch_uncertainty.layers.bayesian_layers import BayesLinear
from torch_uncertainty.losses import ELBOLoss


# fmt: on
class TestELBOLoss:
    """Testing the ELBOLoss class."""

    def test_main(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        with pytest.raises(ValueError):
            ELBOLoss(model, criterion, kl_weight=-1, num_samples=1)

        with pytest.raises(ValueError):
            ELBOLoss(model, criterion, kl_weight=1, num_samples=-1)

        with pytest.raises(TypeError):
            ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1.5)

        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)

        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

    def test_no_bayes(self):
        model = nn.Linear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))
