import math

import pytest
import torch
from torch import nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.losses import BetaNLL, DECLoss, ELBOLoss, NIGLoss


class TestELBOLoss:
    """Testing the ELBOLoss class."""

    def test_main(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)

        loss(model(torch.randn(1, 1)), torch.randn(1, 1))

    def test_failures(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        with pytest.raises(TypeError):
            ELBOLoss(model, nn.BCEWithLogitsLoss, kl_weight=1, num_samples=1)

        with pytest.raises(ValueError):
            ELBOLoss(model, criterion, kl_weight=-1, num_samples=1)

        with pytest.raises(ValueError):
            ELBOLoss(model, criterion, kl_weight=1, num_samples=-1)

        with pytest.raises(TypeError):
            ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1.5)

    def test_no_bayes(self):
        model = nn.Linear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        loss = ELBOLoss(model, criterion, kl_weight=1e-5, num_samples=1)
        loss(model(torch.randn(1, 1)), torch.randn(1, 1))


class TestNIGLoss:
    """Testing the NIGLoss class."""

    def test_main(self):
        loss = NIGLoss(reg_weight=1e-2)

        inputs = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(*inputs.split(1, dim=-1), targets) == pytest.approx(
            2 * math.log(2)
        )

        loss = NIGLoss(
            reg_weight=1e-2,
            reduction="sum",
        )

        assert loss(
            *inputs.repeat(2, 1).split(1, dim=-1),
            targets.repeat(2, 1),
        ) == pytest.approx(4 * math.log(2))

        loss = NIGLoss(
            reg_weight=1e-2,
            reduction="none",
        )

        assert loss(
            *inputs.repeat(2, 1).split(1, dim=-1),
            targets.repeat(2, 1),
        ) == pytest.approx([2 * math.log(2), 2 * math.log(2)])

    def test_failures(self):
        with pytest.raises(ValueError):
            NIGLoss(reg_weight=-1)

        with pytest.raises(ValueError):
            NIGLoss(reg_weight=1.0, reduction="median")


class TestDECLoss:
    """Testing the DECLoss class."""

    def test_main(self):
        loss = DECLoss(
            loss_type="mse", reg_weight=1e-2, annealing_step=1, reduction="sum"
        )
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]), current_epoch=1)
        loss = DECLoss(loss_type="mse", reg_weight=1e-2, annealing_step=1)
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]), current_epoch=0)
        loss = DECLoss(loss_type="log", reg_weight=1e-2, reduction="none")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
        loss = DECLoss(loss_type="digamma")
        loss(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))

    def test_failures(self):
        with pytest.raises(ValueError):
            DECLoss(reg_weight=-1)

        with pytest.raises(ValueError):
            DECLoss(annealing_step=0)

        loss = DECLoss(annealing_step=10)
        with pytest.raises(ValueError):
            loss(
                torch.tensor([[0.0, 0.0]]),
                torch.tensor([0]),
                current_epoch=None,
            )

        with pytest.raises(ValueError):
            DECLoss(reduction="median")

        with pytest.raises(ValueError):
            DECLoss(loss_type="regression")


class TestBetaNLL:
    """Testing the BetaNLL class."""

    def test_main(self):
        loss = BetaNLL(beta=0.5)

        inputs = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0]], dtype=torch.float32)

        assert loss(*inputs.split(1, dim=-1), targets) == 0

        loss = BetaNLL(
            beta=0.5,
            reduction="sum",
        )

        assert (
            loss(
                *inputs.repeat(2, 1).split(1, dim=-1),
                targets.repeat(2, 1),
            )
            == 0
        )

        loss = BetaNLL(
            beta=0.5,
            reduction="none",
        )

        assert loss(
            *inputs.repeat(2, 1).split(1, dim=-1),
            targets.repeat(2, 1),
        ) == pytest.approx([0.0, 0.0])

    def test_failures(self):
        with pytest.raises(ValueError):
            BetaNLL(beta=-1)

        with pytest.raises(ValueError):
            BetaNLL(beta=1.0, reduction="median")
