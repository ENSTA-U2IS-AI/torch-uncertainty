import pytest
import torch

from tests._dummies.model import dummy_model
from torch_uncertainty.models import MCDropout, mc_dropout


class TestMCDropout:
    """Testing the MC Dropout class."""

    def test_mc_dropout_train(self):
        model = dummy_model(10, 5, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10))

        dropout_model = mc_dropout(model, num_estimators=5, last_layer=True)
        dropout_model.train()
        assert dropout_model.training
        dropout_model(torch.rand(1, 10))

    def test_mc_dropout_eval(self):
        model = dummy_model(10, 5, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.eval()
        assert not dropout_model.training
        dropout_model(torch.rand(1, 10))

        dropout_model = mc_dropout(model, num_estimators=5, on_batch=False)
        dropout_model.eval()
        assert not dropout_model.training
        dropout_model(torch.rand(1, 10))

    def test_mc_dropout_errors(self):
        model = dummy_model(10, 5, 0.1)

        with pytest.raises(
            ValueError, match="`num_estimators` must be strictly positive"
        ):
            MCDropout(
                model=model, num_estimators=-1, last_layer=True, on_batch=True
            )

        dropout_model = mc_dropout(model, 5)
        with pytest.raises(
            TypeError, match="Training mode is expected to be boolean"
        ):
            dropout_model.train(mode=1)

        with pytest.raises(
            TypeError, match="Training mode is expected to be boolean"
        ):
            dropout_model.train(mode=None)

        model = dummy_model(10, 5, 0.0)
        with pytest.raises(
            ValueError,
            match="At least one dropout module must have a dropout rate",
        ):
            dropout_model = mc_dropout(model, 5)

        model = dummy_model(10, 5, dropout_rate=0)
        del model.dropout
        with pytest.raises(ValueError):
            dropout_model = mc_dropout(model, 5)
