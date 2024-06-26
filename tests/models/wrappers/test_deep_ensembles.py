import pytest
import torch

from tests._dummies import dummy_model
from torch_uncertainty.models import deep_ensembles


class TestDeepEnsemblesModel:
    """Testing the deep_ensembles function."""

    def test_main(self):
        model_1 = dummy_model(1, 10)
        model_2 = dummy_model(1, 10)

        de = deep_ensembles([model_1, model_2])
        # Check B N C
        assert de(torch.randn(3, 4, 4)).shape == (6, 10)

    def test_list_and_num_estimators(self):
        model_1 = dummy_model(1, 10)
        model_2 = dummy_model(1, 10)
        with pytest.raises(ValueError):
            deep_ensembles([model_1, model_2], num_estimators=2)

    def test_list_singleton(self):
        model_1 = dummy_model(1, 10)

        deep_ensembles([model_1], num_estimators=2, reset_model_parameters=True)
        deep_ensembles(model_1, num_estimators=2, reset_model_parameters=False)

        with pytest.raises(ValueError):
            deep_ensembles([model_1], num_estimators=1)

    def test_errors(self):
        model_1 = dummy_model(1, 10)
        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=None)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=-1)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=1)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=2, task="regression")

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=2, task="other")
