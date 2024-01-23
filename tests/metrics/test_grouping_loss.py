import pytest
import torch

from torch_uncertainty.metrics import GroupingLoss


@pytest.fixture()
def disagreement_probas_3() -> torch.Tensor:
    return torch.as_tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])


class TestGroupingLoss:
    """Testing the GroupingLoss metric class."""

    def test_compute(self):
        metric = GroupingLoss()
        metric.update(
            torch.ones((2, 10)) / 10, torch.arange(2), torch.rand((2, 10))
        )

    def test_errors(self):
        metric = GroupingLoss()
        with pytest.raises(ValueError):
            metric.update(
                torch.ones((2, 10, 1, 1)) / 10,
                torch.arange(2),
                torch.rand((2, 10)),
            )

        with pytest.raises(ValueError):
            metric.update(
                torch.ones((2, 10)) / 10,
                torch.arange(2),
                torch.rand((2, 10, 10, 1, 1)),
            )

        with pytest.raises(ValueError):
            metric.update(
                torch.ones((2, 10)) / 10,
                torch.arange(2).unsqueeze(-1).unsqueeze(-1),
                torch.rand((2, 10, 10, 1, 1)),
            )
