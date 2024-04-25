import pytest
import torch

from torch_uncertainty.metrics import (
    MeanAbsoluteErrorInverse,
    MeanSquaredErrorInverse,
)


class TestMeanAbsoluteErrorInverse:
    """Test the MeanAbsoluteErrorInverse metric."""

    def test_main(self):
        preds = torch.tensor([1, 1 / 2, 1 / 3])
        target = torch.tensor([1, 1 / 2, 1 / 3])
        metric = MeanAbsoluteErrorInverse(unit="m")
        metric.update(preds, target)
        assert metric.compute() == pytest.approx(0)

        metric.reset()
        target = torch.tensor([1, 1, 1])
        metric.update(preds, target)
        assert metric.compute() == pytest.approx(1)

        MeanAbsoluteErrorInverse(unit="mm")
        MeanAbsoluteErrorInverse(unit="km")

    def test_error(self):
        with pytest.raises(ValueError, match="unit must be one of 'mm'"):
            MeanAbsoluteErrorInverse(unit="cm")


class TestMeanSquaredErrorInverse:
    """Test the MeanSquaredErrorInverse metric."""

    def test_main(self):
        preds = torch.tensor([1, 1 / 2, 1 / 3])
        target = torch.tensor([1, 1 / 2, 1 / 3])
        metric = MeanSquaredErrorInverse(unit="m")
        metric.update(preds, target)
        assert metric.compute() == pytest.approx(0)

        metric.reset()
        target = torch.tensor([1, 1, 1])
        metric.update(preds, target)
        assert metric.compute() == pytest.approx(5 / 3)
