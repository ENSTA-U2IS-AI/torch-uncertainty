import pytest
import torch

from torch_uncertainty.metrics import VariationRatio


@pytest.fixture()
def disagreement_probas_3est() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    return torch.as_tensor([[[0.2, 0.8]], [[0.7, 0.3]], [[0.6, 0.4]]])


@pytest.fixture()
def agreement_probas() -> torch.Tensor:
    return torch.as_tensor([[[0.9, 0.1]], [[0.9, 0.1]]])


@pytest.fixture()
def agreement_probas_3est() -> torch.Tensor:
    """Return a vector with mean entropy ~ln(2) and entropy of mean =0."""
    return torch.as_tensor([[[0.2, 0.8]], [[0.3, 0.7]], [[0.4, 0.6]]])


class TestVariationRatio:
    """Testing the VariationRatio metric class."""

    def test_compute_agreement(self, agreement_probas: torch.Tensor):
        metric = VariationRatio(probabilistic=True)
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == pytest.approx(0.1, 1e-6)

        metric = VariationRatio(probabilistic=False)
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

        metric = VariationRatio(probabilistic=False, reduction="none")
        metric.update(agreement_probas)
        res = metric.compute()
        assert res == 0.0

    def test_compute_disagreement(
        self,
        agreement_probas_3est: torch.Tensor,
        disagreement_probas_3est: torch.Tensor,
    ):
        metric = VariationRatio(probabilistic=True, reduction="sum")
        metric.update(disagreement_probas_3est)
        metric.update(agreement_probas_3est)
        res = metric.compute()
        assert res == pytest.approx(0.8, 1e-6)

    def test_bad_argument(self):
        with pytest.raises(ValueError):
            _ = VariationRatio(reduction="geometric_mean")
