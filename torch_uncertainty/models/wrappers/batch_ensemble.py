import torch
from torch import nn


class BatchEnsemble(nn.Module):
    """Wrap a BatchEnsemble model to ensure correct batch replication.

    In a BatchEnsemble architecture, each estimator operates on a **sub-batch**
    of the input. This means that the input batch must be **repeated**
    :attr:`num_estimators` times before being processed.

    This wrapper automatically **duplicates the input batch** along the first axis,
    ensuring that each estimator receives the correct data format.

    **Usage Example:**
    ```python
    model = lenet(in_channels=1, num_classes=10)
    wrapped_model = BatchEnsembleWrapper(model, num_estimators=5)
    logits = wrapped_model(x)  # `x` is automatically repeated `num_estimators` times
    ```

    Args:
        model (nn.Module): The BatchEnsemble model.
        num_estimators (int): Number of ensemble members.
    """

    def __init__(self, model: nn.Module, num_estimators: int):
        super().__init__()
        self.model = model
        self.num_estimators = num_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats the input batch and passes it through the model."""
        repeat_shape = [self.num_estimators] + [1] * (x.dim() - 1)
        x = x.repeat(repeat_shape)
        return self.model(x)
