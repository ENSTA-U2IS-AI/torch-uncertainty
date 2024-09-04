import pytest
import torch

from torch_uncertainty.metrics.classification import FPR95, FPRx


class TestFPR95:
    """Testing the Entropy metric class."""

    def test_compute_zero(self):
        metric = FPR95(pos_label=1)
        metric.update(
            torch.as_tensor([1] * 99 + [0.99]), torch.as_tensor([1] * 99 + [0])
        )
        res = metric.compute()
        assert res == 0

    def test_compute_half(self):
        metric = FPR95(pos_label=1)
        metric.update(
            torch.as_tensor([0.9] * 100 + [0.95] * 50 + [0.85] * 50),
            torch.as_tensor([1] * 100 + [0] * 100),
        )
        res = metric.compute()
        assert res == 0.5

    def test_compute_one(self):
        metric = FPR95(pos_label=1)
        metric.update(
            torch.as_tensor([0.99] * 99 + [1]), torch.as_tensor([1] * 99 + [0])
        )
        res = metric.compute()
        assert res == 1

    def test_compute_nan(self):
        metric = FPR95(pos_label=1)
        metric.update(
            torch.as_tensor([0.1] * 50 + [0.4] * 50), torch.as_tensor([0] * 100)
        )
        res = metric.compute()
        assert torch.isnan(res).all()

    def test_error(self):
        with pytest.raises(ValueError):
            FPRx(recall_level=1.2, pos_label=1)
