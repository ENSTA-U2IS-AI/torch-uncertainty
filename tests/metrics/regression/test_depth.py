import pytest
import torch

from torch_uncertainty.metrics import (
    Log10,
    MeanGTRelativeAbsoluteError,
    MeanGTRelativeSquaredError,
    MeanSquaredLogError,
    SILog,
    ThresholdAccuracy,
)


class TestLog10:
    """Testing the Log10 metric."""

    def test_main(self):
        metric = Log10()
        preds = torch.rand((10, 2)).double() + 0.01
        targets = torch.rand((10, 2)).double() + 0.01
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert torch.mean(
            torch.abs(preds.log10().flatten() - targets.log10().flatten())
        ) == pytest.approx(metric.compute())


class TestMeanGTRelativeAbsoluteError:
    """Testing the MeanGTRelativeAbsoluteError metric."""

    def test_main(self):
        metric = MeanGTRelativeAbsoluteError()
        preds = torch.rand((10, 2))
        targets = torch.rand((10, 2))
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert (torch.abs(preds - targets) / targets).mean() == pytest.approx(
            metric.compute()
        )


class TestMeanGTRelativeSquaredError:
    """Testing the MeanGTRelativeSquaredError metric."""

    def test_main(self):
        metric = MeanGTRelativeSquaredError()
        preds = torch.rand((10, 2))
        targets = torch.rand((10, 2))
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert torch.flatten(
            (preds - targets) ** 2 / targets
        ).mean() == pytest.approx(metric.compute())


class TestSILog:
    """Testing the SILog metric."""

    def test_main(self):
        metric = SILog()
        preds = torch.rand((10, 2)).double()
        targets = torch.rand((10, 2)).double()
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        mean_log_dists = torch.mean(
            targets.flatten().log() - preds.flatten().log()
        )
        assert torch.mean(
            (preds.flatten().log() - targets.flatten().log() + mean_log_dists)
            ** 2
        ) == pytest.approx(metric.compute())

        metric = SILog(sqrt=True)
        preds = torch.rand((10, 2)).double()
        targets = torch.rand((10, 2)).double()
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        mean_log_dists = torch.mean(
            targets.flatten().log() - preds.flatten().log()
        )
        assert torch.mean(
            (preds.flatten().log() - targets.flatten().log() + mean_log_dists)
            ** 2
        ) ** 0.5 == pytest.approx(metric.compute())


class TestThresholdAccuracy:
    """Testing the ThresholdAccuracy metric."""

    def test_main(self):
        metric = ThresholdAccuracy(power=1, lmbda=1.25)
        preds = torch.ones((10, 2))
        targets = torch.ones((10, 2)) * 1.3
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert metric.compute() == 0.0

        metric = ThresholdAccuracy(power=1, lmbda=1.25)
        preds = torch.cat(
            [torch.ones((10, 2)) * 1.2, torch.ones((10, 2))], dim=0
        )
        targets = torch.ones((20, 2)) * 1.3
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert metric.compute() == 0.5

    def test_error(self):
        with pytest.raises(ValueError, match="Power must be"):
            ThresholdAccuracy(power=-1)
        with pytest.raises(ValueError, match="Lambda must be"):
            ThresholdAccuracy(power=1, lmbda=0.5)


class TestMeanSquaredLogError:
    """Testing the MeanSquaredLogError metric."""

    def test_main(self):
        metric = MeanSquaredLogError()
        preds = torch.rand((10, 2)).double()
        targets = torch.rand((10, 2)).double()
        metric.update(preds[:, 0], targets[:, 0])
        metric.update(preds[:, 1], targets[:, 1])
        assert torch.mean(
            (preds.log() - targets.log()).flatten() ** 2
        ) == pytest.approx(metric.compute())
