import pytest
import torch

from torch_uncertainty.metrics import Disagreement


@pytest.fixture()
def disagreement_probas() -> torch.Tensor:
    return torch.as_tensor([[[0.0, 1.0], [1.0, 0.0]]])


@pytest.fixture()
def agreement_probas() -> torch.Tensor:
    return torch.as_tensor([[[1.0, 0.0], [1.0, 0.0]]])


@pytest.fixture()
def disagreement_probas_3() -> torch.Tensor:
    return torch.as_tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])


class TestDisagreement:
    """Testing the Disagreement metric class."""

    def test_compute_disagreement(self, disagreement_probas: torch.Tensor):
        metric = Disagreement()
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == 1.0

    def test_compute_agreement(self, agreement_probas: torch.Tensor):
        metric = Disagreement()
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

    def test_compute_mixed(
        self, disagreement_probas: torch.Tensor, agreement_probas: torch.Tensor
    ):
        metric = Disagreement()
        metric.update(agreement_probas)
        metric.update(disagreement_probas)
        res = metric.compute()
        assert res == 0.5

    def test_compute_mixed_3_estimators(
        self, disagreement_probas_3: torch.Tensor
    ):
        metric = Disagreement()
        metric.update(disagreement_probas_3)
        res = metric.compute()
        assert res == pytest.approx(2 / 3, 1e-6)

        metric = Disagreement(reduction="sum")
        metric.update(disagreement_probas_3)
        res = metric.compute()
        assert res == pytest.approx(2 / 3, 1e-6)

        metric = Disagreement(reduction="none")
        metric.update(disagreement_probas_3)
        res = metric.compute()
        assert res == pytest.approx(2 / 3, 1e-6)

    def test_bad_argument_reduction(self):
        with pytest.raises(ValueError):
            _ = Disagreement(reduction="geometric_mean")
