import pytest

from torch_uncertainty.baselines.classification.deep_ensembles import (
    DeepEnsemblesBaseline,
)


class TestDeepEnsembles:
    """Testing the Deep Ensembles baseline class."""

    def test_failure(self):
        with pytest.raises(
            ValueError, match="Models must not be an empty list."
        ):
            DeepEnsemblesBaseline(
                log_path=".",
                checkpoint_ids=[],
                backbone="resnet",
                num_classes=10,
            )
