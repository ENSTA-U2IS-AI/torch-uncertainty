import pytest
import torch
from torch import nn, optim

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.layers.distributions import NormalLinear
from torch_uncertainty.losses import DistributionNLLLoss, ELBOLoss
from torch_uncertainty.routines import RegressionRoutine


class TestELBOLoss:
    """Testing the ELBOLoss class."""

    def test_main(self) -> None:
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()
        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

        model = nn.Linear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        ELBOLoss(None, criterion, kl_weight=1e-5, num_samples=1)
        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

    def test_prob_regression_training_step(self) -> None:
        model = NormalLinear(BayesLinear, event_dim=4, in_features=10)
        criterion = DistributionNLLLoss()
        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=3, dist_family="normal")

        routine = RegressionRoutine(
            output_dim=1,
            model=model,
            loss=loss,
            dist_family="normal",
            optim_recipe=optim.Adam(
                model.parameters(),
                lr=5e-4,
                weight_decay=0,
            ),
        )
        inputs = torch.randn(1, 10)
        targets = torch.randn(1, 4)
        routine.training_step((inputs, targets))

    def test_training_step(self) -> None:
        model = BayesLinear(10, 4)
        criterion = nn.MSELoss()
        loss = ELBOLoss(model, criterion, kl_weight=1 / 50000, num_samples=3)

        routine = RegressionRoutine(
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
        routine.training_step((inputs, targets))

    def test_failures(self) -> None:
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        with pytest.raises(TypeError, match="The inner_loss should be an instance of a class."):
            ELBOLoss(model, nn.BCEWithLogitsLoss, kl_weight=1, num_samples=1)

        with pytest.raises(ValueError, match="The KL weight should be non-negative. Got "):
            ELBOLoss(model, criterion, kl_weight=-1, num_samples=1)

        with pytest.raises(
            ValueError,
            match="The number of samples should not be lower than 1.",
        ):
            ELBOLoss(model, criterion, kl_weight=1, num_samples=-1)

        with pytest.raises(TypeError, match="The number of samples should be an integer. "):
            ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1.5)
