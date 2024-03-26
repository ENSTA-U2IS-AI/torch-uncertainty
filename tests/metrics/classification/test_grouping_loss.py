import pytest
import torch

from torch_uncertainty.metrics import GroupingLoss


class TestGroupingLoss:
    """Testing the GroupingLoss metric class."""

    def test_compute(self):
        metric = GroupingLoss()
        metric.update(
            torch.cat([torch.tensor([0, 1, 0, 1]), torch.ones(200) / 10]),
            torch.cat(
                [torch.tensor([0, 0, 1, 1]), torch.zeros(100), torch.ones(100)]
            ).long(),
            torch.cat([torch.zeros((104, 10)), torch.ones((100, 10))]),
        )
        metric.compute()

        metric = GroupingLoss()
        metric.update(
            torch.ones((200, 4, 10)),
            torch.cat([torch.arange(100), torch.arange(100)]),
            torch.cat([torch.zeros((100, 4, 10)), torch.ones((100, 4, 10))]),
        )
        metric.compute()
        metric.reset()

        metric.update(
            torch.ones((200, 10)) / 10,
            torch.nn.functional.one_hot(torch.arange(200)),
            torch.cat([torch.zeros((100, 10)), torch.ones((1004, 10))]),
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
