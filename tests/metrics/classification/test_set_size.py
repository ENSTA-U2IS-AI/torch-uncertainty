import pytest
import torch

from torch_uncertainty.metrics import SetSize


class TestCoverageRate:
    """Testing the CoverageRate metric class."""

    def test_main(self) -> None:
        metric = SetSize()
        pred = torch.tensor([0.2, 0.2, 0.0, 0.2, 0.2]).unsqueeze(0)
        assert metric(pred) == 4

        metric = SetSize(reduction=None)
        pred = torch.tensor([[0.2, 0.2, 0.2, 0.0, 0.2], [0.2, 0.0, 0.0, 0.0, 0.2]]).repeat(2, 1)
        assert all(metric(pred) == torch.tensor([4, 2, 4, 2]))

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError, match="Expected argument"):
            SetSize(reduction="42")
