from torch_uncertainty.baselines.classification import DeepEnsembles


class TestDeepEnsembles:
    """Testing the Deep Ensembles baseline class."""

    def test_standard(self):
        DeepEnsembles(
            log_path=".",
            checkpoint_ids=[],
            backbone="resnet",
            num_classes=10,
        )
        # parser = ArgumentParser()
        # DeepEnsembles.add_model_specific_args(parser)
