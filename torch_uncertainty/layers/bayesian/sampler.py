import numpy as np
import torch
from torch import Tensor, distributions, nn


class TrainableDistribution(nn.Module):
    lsqrt2pi = torch.tensor(np.log(np.sqrt(2 * np.pi)))

    def __init__(
        self,
        mu: Tensor,
        rho: Tensor,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = None
        self.weight = None

    def sample(self) -> Tensor:
        w_sample = torch.randn(size=self.mu.shape, device=self.mu.device)
        self.sigma = torch.log1p(torch.exp(self.rho)).to(self.mu.device)
        self.weight = self.mu + self.sigma * w_sample
        return self.weight

    def log_posterior(self, weight: Tensor | None = None) -> Tensor:
        if self.weight is None or self.sigma is None:
            raise ValueError(
                "Sample the weights before querying the log posterior."
            )

        if weight is None:  # coverage: ignore
            weight = self.weight

        lposterior = (
            self.lsqrt2pi
            + torch.log(self.sigma)
            + (((weight - self.mu) ** 2) / (2 * self.sigma**2))
            + 0.5
        )
        return -lposterior.sum()


class CenteredGaussianMixture(nn.Module):
    def __init__(
        self,
        sigma_1: float,
        sigma_2: float,
        pi: float,
    ) -> None:
        """Create a mixture of two centered Gaussian distributions.

        Args:
            sigma_1 (float): Standard deviation of the first Gaussian.
            sigma_2 (float): Standard deviation of the second Gaussian.
            pi (float): Mixing coefficient.
        """
        super().__init__()
        self.register_buffer("pi", torch.tensor([pi, 1 - pi]))
        self.register_buffer("mus", torch.zeros(2))
        self.register_buffer("sigmas", torch.tensor([sigma_1, sigma_2]))

    def log_prob(self, weight: Tensor) -> Tensor:
        mix = distributions.Categorical(self.pi)
        normals = distributions.Normal(self.mus, self.sigmas)
        distribution = distributions.MixtureSameFamily(mix, normals)
        return distribution.log_prob(weight).sum()
