import pytest
import torch

from torch_uncertainty.metrics import CoverageRate


class TestCoverageRate:
    """Testing the CoverageRate metric class."""

    def test_main(self) -> None:
        metric = CoverageRate()

        preds = torch.tensor(
            [
                [True, True, False],
                [False, False, True],
                [True, True, False],
                [False, True, False],
                [True, True, False],
                [True, True, False],
                [False, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
            ]
        )
        labels = torch.tensor([0] * 10)
        metric.update(preds, labels)
        assert metric.compute() == pytest.approx(0.7, rel=1e-2)

        metric = CoverageRate(validate_args=False)
        labels = torch.tensor([1] * 10)
        metric.update(preds, labels)
        assert metric.compute() == pytest.approx(0.9, rel=1e-2)
        metric.reset()
        labels = torch.tensor([2] * 10)
        metric.update(preds, labels)
        assert metric.compute() == pytest.approx(0.1, rel=1e-2)

        metric = CoverageRate(num_classes=3, average="macro")
        labels = torch.tensor([0] * 3 + [1] * 3 + [2] * 4)
        metric.update(preds, labels)
        assert metric.compute() == pytest.approx(0.5556, rel=1e-2)

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            CoverageRate(num_classes=1)
        with pytest.raises(ValueError):
            CoverageRate(num_classes=3, average="invalid")
        with pytest.raises(ValueError):
            CoverageRate(num_classes=None, average="macro")
