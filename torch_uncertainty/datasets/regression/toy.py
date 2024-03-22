import torch
from torch.utils.data import TensorDataset


class Cubic(TensorDataset):
    """A dataset of samples drawn from the cube fct. with homoscedastic noise.

    Args:
        lower_bound (float, optional): Lower bound of the samples. Defaults to
            ``-4.0``.
        upper_bound (float, optional): Upper bound of the samples. Defaults to
            ``4.0``.
        num_samples (int, optional): Number of samples. Defaults to ``5000``.
        noise_mean (float, optional): Mean of the noise. Defaults to ``0.0``.
        noise_std (float, optional): Standard deviation of the noise. Defaults
            to ``3.0``.
    """

    def __init__(
        self,
        lower_bound: float = -4.0,
        upper_bound: float = 4.0,
        num_samples: int = 5000,
        noise_mean: float = 0.0,
        noise_std: float = 3.0,
    ) -> None:
        noise = (noise_mean, noise_std)

        samples = torch.linspace(
            lower_bound, upper_bound, num_samples
        ).unsqueeze(-1)
        targets = samples**3 + torch.normal(*noise, size=samples.size())
        super().__init__(samples, targets.squeeze(-1))
