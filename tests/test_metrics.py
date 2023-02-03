# fmt:off

import pytest
import torch
from torch_uncertainty.metrics import Entropy
import math
# fmt:on


@pytest.fixture
def vec_max():
    vec = torch.as_tensor([0.5, 0.5])
    return vec.unsqueeze(0)


@pytest.fixture
def vec_min():
    vec = torch.as_tensor([0, 1])
    return vec.unsqueeze(0)


class TestEntropy:
    def test_init(self):
        self.entropy = Entropy()

    def test_update(self, vec_min):
        self.entropy = Entropy()
        self.entropy.update(vec_min)

    def test_multiple_updates(self, vec_min, vec_max):
        self.entropy = Entropy()
        self.entropy.update(vec_min)
        self.entropy.update(vec_max)

    def test_compute(self, vec_min):
        self.entropy = Entropy()
        self.entropy.update(vec_min)
        res = self.entropy.compute()
        assert res == 0

    def test_compute_max(self, vec_max):
        self.entropy = Entropy(reduction="sum")
        self.entropy.update(vec_max)
        res = self.entropy.compute()
        assert res == math.log(2)
