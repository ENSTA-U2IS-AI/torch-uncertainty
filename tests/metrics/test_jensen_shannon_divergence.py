# fmt:off
import math

import pytest
import torch

from torch_uncertainty.metrics import JensenShannonDivergence


# fmt:on
@pytest.fixture
def disagreement_probas() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    vec = torch.as_tensor([[[1e-8, 1 - 1e-8], [1 - 1e-8, 1e-8]]])
    print(vec.shape)
    return vec


@pytest.fixture
def agreement_probas() -> torch.Tensor:
    vec = torch.as_tensor([[[0.9, 0.1], [0.9, 0.1]]])
    return vec


class TestJensenShannonDivergence:
    """Testing the JensenShannonDivergence metric class."""

    def test_compute_disagreement(self, disagreement_probas: torch.Tensor):
        metric = JensenShannonDivergence()
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == pytest.approx(math.log(2), 1e-5)

    def test_compute_agreement(self, agreement_probas: torch.Tensor):
        metric = JensenShannonDivergence()
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

    def test_compute_mixed(
        self, disagreement_probas: torch.Tensor, agreement_probas: torch.Tensor
    ):
        metric = JensenShannonDivergence()
        metric.update(agreement_probas)
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == pytest.approx(math.log(2) / 2, 1e-5)

    def test_bad_argument(self, agreement_probas: torch.Tensor):
        with pytest.raises(Exception):
            metric = JensenShannonDivergence()
            metric.update(agreement_probas)
            res = metric.compute()
            assert res == 0.5
