import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tests._dummies.model import dummy_model
from torch_uncertainty.models import SWA, SWAG


class TestSWA:
    """Testing the SWA class."""

    def test_training(self):
        dl = DataLoader(TensorDataset(torch.randn(1, 1)), batch_size=1)
        swa = SWA(dummy_model(1, 10), cycle_start=1, cycle_length=1)
        swa.eval()
        swa(torch.randn(1, 1))

        swa.train()
        swa(torch.randn(1, 1))
        swa.update_model(0)
        swa.update_bn(dl, "cpu")

        swa.update_model(1)
        swa.update_bn(dl, "cpu")

        swa.eval()
        swa(torch.randn(1, 1))

    def test_failures(self):
        with pytest.raises(
            ValueError, match="`cycle_start` must be non-negative."
        ):
            SWA(nn.Module(), cycle_start=-1, cycle_length=1)
        with pytest.raises(
            ValueError, match="`cycle_length` must be strictly positive."
        ):
            SWA(nn.Module(), cycle_start=1, cycle_length=0)


class TestSWAG:
    """Testing the SWAG class."""

    def test_training(self):
        dl = DataLoader(TensorDataset(torch.randn(1, 1)), batch_size=1)
        swag = SWAG(
            dummy_model(1, 10), cycle_start=1, cycle_length=1, max_num_models=3
        )
        swag.eval()
        swag(torch.randn(1, 1))

        swag.train()
        swag(torch.randn(1, 1))
        swag.update_model(0)
        swag.update_bn(dl, "cpu")
        swag(torch.randn(1, 1))

        swag.update_model(1)
        swag.update_bn(dl, "cpu")

        swag.update_model(2)
        swag.update_bn(dl, "cpu")
        swag(torch.randn(1, 1))
        swag.update_model(3)
        swag.update_model(4)

        swag.eval()
        swag(torch.randn(1, 1))

        swag = SWAG(
            dummy_model(1, 10),
            cycle_start=1,
            cycle_length=1,
            diag_covariance=True,
        )
        swag.train()
        swag.update_model(2)
        swag.sample(1, True, False, seed=1)

    def test_state_dict(self):
        mod = dummy_model(1, 10)
        swag = SWAG(mod, cycle_start=1, cycle_length=1, num_estimators=3)
        print(swag.state_dict())
        swag.load_state_dict(swag.state_dict())

    def test_failures(self):
        with pytest.raises(
            NotImplementedError, match="Raise an issue if you need this feature"
        ):
            swag = SWAG(nn.Module(), scale=1, cycle_start=1, cycle_length=1)
            swag.sample(scale=1, block=True)
        with pytest.raises(ValueError, match="`scale` must be non-negative."):
            SWAG(nn.Module(), scale=-1, cycle_start=1, cycle_length=1)
        with pytest.raises(
            ValueError, match="`max_num_models` must be non-negative."
        ):
            SWAG(nn.Module(), max_num_models=-1, cycle_start=1, cycle_length=1)
        with pytest.raises(
            ValueError, match="`var_clamp` must be non-negative. "
        ):
            SWAG(nn.Module(), var_clamp=-1, cycle_start=1, cycle_length=1)
        swag = SWAG(
            nn.Module(), cycle_start=1, cycle_length=1, diag_covariance=True
        )
        with pytest.raises(
            ValueError,
            match="Cannot sample full rank from diagonal covariance matrix.",
        ):
            swag.sample(scale=1, diag_covariance=False)
