import pytest
import torch
from torch import nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.losses import (
    ELBOLoss,
)


class TestELBOLoss:
    """Testing the ELBOLoss class."""

    def test_main(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()
        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

        model = nn.Linear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        ELBOLoss(None, criterion, kl_weight=1e-5, num_samples=1)
        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

    def test_failures(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        with pytest.raises(
            TypeError, match="The inner_loss should be an instance of a class."
        ):
            ELBOLoss(model, nn.BCEWithLogitsLoss, kl_weight=1, num_samples=1)

        with pytest.raises(
            ValueError, match="The KL weight should be non-negative. Got "
        ):
            ELBOLoss(model, criterion, kl_weight=-1, num_samples=1)

        with pytest.raises(
            ValueError,
            match="The number of samples should not be lower than 1.",
        ):
            ELBOLoss(model, criterion, kl_weight=1, num_samples=-1)

        with pytest.raises(
            TypeError, match="The number of samples should be an integer. "
        ):
            ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1.5)
