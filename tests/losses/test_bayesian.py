import pytest
import torch
from torch import nn, optim

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.routines import RegressionRoutine


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

    def test_training_step(self):
        model = BayesLinear(10, 4)
        criterion = nn.MSELoss()
        loss = ELBOLoss(model, criterion, kl_weight=1 / 50000, num_samples=3)

        routine = RegressionRoutine(
            probabilistic=False,
            output_dim=4,
            model=model,
            loss=loss,
            optim_recipe=optim.Adam(
                model.parameters(),
                lr=5e-4,
                weight_decay=0,
            ),
        )

        inputs = torch.randn(1, 10)
        targets = torch.randn(1, 4)
        routine.training_step((inputs, targets), 0)

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
