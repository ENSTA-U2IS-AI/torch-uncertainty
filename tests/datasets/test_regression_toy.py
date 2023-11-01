from torch_uncertainty.datasets.regression.toy import Cubic


class TestCubic:
    """Testing the Cubic dataset class."""

    def test_main(self):
        ds = Cubic(num_samples=10)
        _ = ds[9]
        _ = len(ds)
