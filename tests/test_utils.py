from pathlib import Path

import pytest
import torch
from torch.distributions import Laplace, Normal

from torch_uncertainty.utils import distributions, get_version, hub


class TestUtils:
    """Testing utils methods."""

    def test_get_version_log_success(self):
        get_version("tests/testlog", version=42)
        get_version(Path("tests/testlog"), version=42)

        get_version("tests/testlog", version=42, checkpoint=45)

    def test_getversion_log_failure(self):
        with pytest.raises(Exception):
            get_version("tests/testlog", version=52)


class TestHub:
    """Testing hub methods."""

    def test_hub_exists(self):
        hub.load_hf("test")
        hub.load_hf("test", version=1)
        hub.load_hf("test", version=2)

    def test_hub_notexists(self):
        with pytest.raises(Exception):
            hub.load_hf("tests")

        with pytest.raises(ValueError):
            hub.load_hf("test", version=42)


class TestDistributions:
    """Testing distributions methods."""

    def test_nig(self):
        dist = distributions.NormalInverseGamma(
            torch.tensor(0.0),
            torch.tensor(1.1),
            torch.tensor(1.1),
            torch.tensor(1.1),
        )
        _ = dist.mean, dist.mean_loc, dist.mean_variance, dist.variance_loc

    def test_errors(self):
        with pytest.raises(ValueError):
            distributions.cat_dist(
                [
                    Normal(torch.tensor([0.0]), torch.tensor([1.0])),
                    Laplace(torch.tensor([0.0]), torch.tensor([1.0])),
                ],
                dim=0,
            )
