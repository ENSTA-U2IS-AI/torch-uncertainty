import pytest
import torch
from einops import repeat
from torch import nn
from torch.utils.data import DataLoader

from torch_uncertainty.post_processing import (
    Conformal,
    ConformalClsAPS,
    ConformalClsRAPS,
    ConformalClsTHR,
)


class TestConformal:
    """Testing the Conformal class."""

    def test_errors(self):
        Conformal.__abstractmethods__ = set()
        conformal = Conformal(model=None)
        assert conformal.model is None
        conformal.set_model(nn.Identity())
        assert isinstance(conformal.model, nn.Identity)
        conformal.fit(None)
        conformal.forward(None)
        conformal.conformal(None)


class TestConformalClsAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self):
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10)
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsAPS(alpha=0.1, model=nn.Identity(), randomized=True)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_failures(self):
        with pytest.raises(NotImplementedError):
            _ = ConformalClsAPS(alpha=0.1, score_type="test")

        with pytest.raises(ValueError):
            _ = ConformalClsAPS(
                alpha=0.1,
            ).quantile


class TestConformalClsRAPS:
    """Testing the ConformalClsRAPS class."""

    def test_fit(self):
        inputs = repeat(torch.tensor([6.0, 4.0, 1.0]), "c -> b c", b=10)
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsRAPS(alpha=0.1, model=nn.Identity(), randomized=False)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

        conformal = ConformalClsRAPS(alpha=0.1, model=nn.Identity(), randomized=True)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)

    def test_failures(self):
        with pytest.raises(NotImplementedError):
            ConformalClsRAPS(alpha=0.1, score_type="test")

        with pytest.raises(ValueError):
            ConformalClsRAPS(alpha=0.1).quantile  # noqa: B018


class TestConformalClsTHR:
    """Testing the ConformalClsTHR class."""

    def test_main(self):
        conformal = ConformalClsTHR(alpha=0.1, model=None, init_val=2)

        assert conformal.temperature == 2.0

        conformal.set_model(nn.Identity())

        assert isinstance(conformal.model, nn.Identity)
        assert isinstance(conformal.temperature_scaler.model, nn.Identity)

    def test_fit(self):
        inputs = repeat(torch.tensor([0.6, 0.3, 0.1]), "c -> b c", b=10)
        labels = torch.tensor([0, 2] + [1] * 8)

        calibration_set = list(zip(inputs, labels, strict=True))
        dl = DataLoader(calibration_set, batch_size=10)

        conformal = ConformalClsTHR(alpha=0.1, model=nn.Identity(), init_val=2, lr=1, max_iter=10)
        conformal.fit(dl)
        out = conformal.conformal(inputs)
        assert out.shape == (10, 3)
        assert (
            out == repeat(torch.tensor([True, True, False]), "c -> b c", b=10).float() / 2
        ).all()

    def test_failures(self):
        with pytest.raises(ValueError):
            _ = ConformalClsTHR(
                alpha=0.1,
            ).quantile
