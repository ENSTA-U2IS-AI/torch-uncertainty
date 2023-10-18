# fmt: off
import math

import pytest
import torch
from torch import nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.losses import ELBOLoss, NIGLoss


# fmt: on
class TestELBOLoss:
    """Testing the ELBOLoss class."""

    def test_main(self):
        model = BayesLinear(1, 1)
        criterion = nn.BCEWithLogitsLoss()

        with pytest.raises(ValueError):
            ELBOLoss(model, nn.BCEWithLogitsLoss, kl_weight=1, num_samples=1)

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


# fmt: on
class TestNIGLoss:
    def test_main(self):
        with pytest.raises(ValueError):
            NIGLoss(reg_weight=-1)

        with pytest.raises(ValueError):
            NIGLoss(reg_weight=1.0, reduction="tttt")

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
