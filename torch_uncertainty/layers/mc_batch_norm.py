import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


class _MCBatchNorm(_BatchNorm):
    counter: int

    def __init__(
        self,
        num_features: int,
        num_estimators: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype,
        )

        if num_estimators < 1 or not isinstance(num_estimators, int):
            raise ValueError(
                "num_estimators should be an integer greater or equal than 1. "
                f"got {num_estimators}."
            )

        self.register_buffer(
            "means",
            torch.zeros(
                num_estimators, num_features, device=device, dtype=dtype
            ),
        )
        self.register_buffer(
            "vars",
            torch.zeros(
                num_estimators, num_features, device=device, dtype=dtype
            ),
        )

        self.accumulate = True
        self.num_estimators = num_estimators
        self.reset_mc_statistics()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.
        """
        if not self.training:
            if self.accumulate:
                mean = x.mean((0, -2, -1))
                var = x.var((0, -2, -1), unbiased=True)
                self.means[self.counter] = mean
                self.vars[self.counter] = var
            self.running_mean = self.means[self.counter]
            self.running_var = self.vars[self.counter]
        return super().forward(x)

    def set_counter(self, counter: int) -> None:
        """Set the counter.

        Args:
            counter (int): new value for the counter.
        """
        self.counter = counter % self.num_estimators

    def reset_mc_statistics(self) -> None:
        """Reset the batch statistics."""
        self.counter = 0
        self.means = torch.zeros_like(self.means)
        self.vars = torch.zeros_like(self.vars)


class MCBatchNorm1d(_MCBatchNorm):
    """Monte Carlo Batch Normalization over a 2D or 3D (batched) input.

    Args:
        num_features (int): Number of features.
        num_estimators (int): Number of estimators.
        eps (float, optional): Epsilon. Defaults to 0.00001.
        momentum (float, optional): Momentum. Defaults to 0.1.
        affine (bool, optional): Affine. Defaults to True.
        track_running_stats (bool, optional): Track running statistics.
            Defaults to True.
        device (optional): Device. Defaults to None.
        dtype (optional): Data type. Defaults to None.

    Warning:
        This layer should not be used out of the corresponding wrapper.
        Check MCBatchNorm in torch_uncertainty/post_processing/.
    """

    def _check_input_dim(self, inputs) -> None:
        if inputs.dim() != 2 and inputs.dim() != 3:
            raise ValueError(
                f"expected 2D or 3D input (got {inputs.dim()}D input)"
            )


class MCBatchNorm2d(_MCBatchNorm):
    """Monte Carlo Batch Normalization over a 3D or 4D (batched) input.

    Args:
        num_features (int): Number of features.
        num_estimators (int): Number of estimators.
        eps (float, optional): Epsilon. Defaults to 0.00001.
        momentum (float, optional): Momentum. Defaults to 0.1.
        affine (bool, optional): Affine. Defaults to True.
        track_running_stats (bool, optional): Track running statistics.
            Defaults to True.
        device (optional): Device. Defaults to None.
        dtype (optional): Data type. Defaults to None.

    Warning:
        This layer should not be used out of the corresponding wrapper.
        Check MCBatchNorm in torch_uncertainty/post_processing/.
    """

    def _check_input_dim(self, inputs) -> None:
        if inputs.dim() != 3 and inputs.dim() != 4:
            raise ValueError(
                f"expected 3D or 4D input (got {inputs.dim()}D input)"
            )


class MCBatchNorm3d(_MCBatchNorm):
    """Monte Carlo Batch Normalization over a 4D or 5D (batched) input.

    Args:
        num_features (int): Number of features.
        num_estimators (int): Number of estimators.
        eps (float, optional): Epsilon. Defaults to 0.00001.
        momentum (float, optional): Momentum. Defaults to 0.1.
        affine (bool, optional): Affine. Defaults to True.
        track_running_stats (bool, optional): Track running statistics.
            Defaults to True.
        device (optional): Device. Defaults to None.
        dtype (optional): Data type. Defaults to None.

    Warning:
        This layer should not be used out of the corresponding wrapper.
        Check MCBatchNorm in torch_uncertainty/post_processing/.
    """

    def _check_input_dim(self, inputs) -> None:
        if inputs.dim() != 4 and inputs.dim() != 5:
            raise ValueError(
                f"expected 4D or 5D input (got {inputs.dim()}D input)"
            )
