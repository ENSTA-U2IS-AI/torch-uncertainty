import pytest
import torch

from tests._dummies import dummy_model
from torch_uncertainty.models import deep_ensembles


class TestDeepEnsemblesModel:
    """Testing the deep_ensembles function."""

    def test_main(self):
        model_1 = dummy_model(1, 10, 1)
        model_2 = dummy_model(1, 10, 1)

        de = deep_ensembles([model_1, model_2])
        # Check B N C
        assert de(torch.randn(3, 4, 4)).shape == (3, 2, 10)

    def test_list_and_num_estimators(self):
        model_1 = dummy_model(1, 10, 1)
        model_2 = dummy_model(1, 10, 1)
        with pytest.raises(ValueError):
            deep_ensembles([model_1, model_2], num_estimators=2)

    def test_list_singleton(self):
        model_1 = dummy_model(1, 10, 1)

        deep_ensembles([model_1], num_estimators=2)
        deep_ensembles(model_1, num_estimators=2)

        with pytest.raises(ValueError):
            deep_ensembles([model_1], num_estimators=1)

    def test_model_and_no_num_estimator(self):
        model_1 = dummy_model(1, 10, 1)
        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=None)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=-1)

        with pytest.raises(ValueError):
            deep_ensembles(model_1, num_estimators=1)
