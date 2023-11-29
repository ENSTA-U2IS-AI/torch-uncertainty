import pytest
import torch

from torch_uncertainty.metrics import FPR95


@pytest.fixture()
def confs_zero() -> torch.Tensor:
    return torch.as_tensor([1] * 99 + [0.99])


@pytest.fixture()
def target_zero() -> torch.Tensor:
    return torch.as_tensor([1] * 99 + [0])


@pytest.fixture()
def confs_half() -> torch.Tensor:
    return torch.as_tensor([0.9] * 100 + [0.95] * 50 + [0.85] * 50)


@pytest.fixture()
def target_half() -> torch.Tensor:
    return torch.as_tensor([1] * 100 + [0] * 100)


@pytest.fixture()
def confs_one() -> torch.Tensor:
    return torch.as_tensor([0.99] * 99 + [1])


@pytest.fixture()
def target_one() -> torch.Tensor:
    return torch.as_tensor([1] * 99 + [0])


class TestFPR95:
    """Testing the Entropy metric class."""

    def test_compute_zero(
        self, confs_zero: torch.Tensor, target_zero: torch.Tensor
    ):
        metric = FPR95(pos_label=1)
        metric.update(confs_zero, target_zero)
        res = metric.compute()
        assert res == 0

    def test_compute_half(
        self, confs_half: torch.Tensor, target_half: torch.Tensor
    ):
        metric = FPR95(pos_label=1)
        metric.update(confs_half, target_half)
        res = metric.compute()
        assert res == 0.5

    def test_compute_one(
        self, confs_one: torch.Tensor, target_one: torch.Tensor
    ):
        metric = FPR95(pos_label=1)
        metric.update(confs_one, target_one)
        res = metric.compute()
        assert res == 1
