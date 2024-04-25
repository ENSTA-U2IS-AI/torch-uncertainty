import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric


# Custom Metric for iMAE
class InverseMAE(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        assert preds.shape == target.shape
        self.total += torch.sum(torch.reciprocal(torch.abs(target - preds)))
        self.count += target.numel()

    def compute(self):
        return self.total / self.count


# Custom Metric for iRMSE
class InverseRMSE(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.mse = MeanSquaredError()

    def update(self, preds: Tensor, target: Tensor):
        self.mse.update(preds, target)

    def compute(self):
        mse_val = self.mse.compute()
        return torch.reciprocal(torch.sqrt(mse_val))
