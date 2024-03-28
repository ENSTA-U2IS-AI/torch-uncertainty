import math

import pytest
import torch

from torch_uncertainty.metrics import MutualInformation


@pytest.fixture()
def disagreement_probas() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    return torch.as_tensor([[[1e-8, 1 - 1e-8], [1 - 1e-8, 1e-8]]])


@pytest.fixture()
def agreement_probas() -> torch.Tensor:
    return torch.as_tensor([[[0.9, 0.1], [0.9, 0.1]]])


class TestMutualInformation:
    """Testing the MutualInformation metric class."""

    def test_compute_disagreement(self, disagreement_probas: torch.Tensor):
        metric = MutualInformation()
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == pytest.approx(math.log(2), 1e-5)

    def test_compute_agreement(self, agreement_probas: torch.Tensor):
        metric = MutualInformation()
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

    def test_compute_mixed(
        self, disagreement_probas: torch.Tensor, agreement_probas: torch.Tensor
    ):
        metric = MutualInformation(reduction="mean")
        metric.update(agreement_probas)
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == pytest.approx(math.log(2) / 2, 1e-5)

        metric = MutualInformation(reduction="sum")
        metric.update(agreement_probas)
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == pytest.approx(math.log(2), 1e-5)

        metric = MutualInformation(reduction=None)
        metric.update(agreement_probas)
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res[0] == pytest.approx(0, 1e-5)
        assert res[1] == pytest.approx(math.log(2), 1e-5)

    def test_bad_argument(self):
        with pytest.raises(ValueError):
            _ = MutualInformation("geometric_mean")
