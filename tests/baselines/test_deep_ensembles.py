from argparse import ArgumentParser

from torch_uncertainty.baselines import DeepEnsembles


class TestDeepEnsembles:
    """Testing the Deep Ensembles baseline class."""

    def test_standard(self):
        DeepEnsembles(
            task="classification",
            log_path=".",
            checkpoint_ids=[],
            backbone="resnet",
            in_channels=3,
            num_classes=10,
            version="std",
            arch=18,
            style="cifar",
            groups=1,
        )
        parser = ArgumentParser()
        DeepEnsembles.add_model_specific_args(parser)
