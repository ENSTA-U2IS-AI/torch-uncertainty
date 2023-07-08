# fmt: off
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np


# fmt: on
class TrainableDistribution(nn.Module):
    lsqrt2pi = torch.as_tensor(np.log(np.sqrt(2 * np.pi)))

    def __init__(self, mu: Tensor, rho: Tensor) -> None:
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.sigma = None
        self.weight = None

    def sample(self) -> Tensor:
        w_sample = torch.normal(mean=0, std=1, size=self.mu.shape)
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.weight = self.mu + self.sigma * w_sample
        return self.weight

    def log_posterior(self, weight: Optional[Tensor] = None) -> Tensor:
        if self.weight is None or self.sigma is None:
            raise ValueError(
                "Sample the weights before asking for the log posterior."
            )

        if weight is None:
            weight = self.weight

        lposterior = (
            self.lsqrt2pi
            + torch.log(self.sigma)
            + (((weight - self.mu) ** 2) / (2 * self.sigma**2))
            + 0.5
        )
        return -lposterior.sum()


class PriorDistribution(nn.Module):
    def __init__(self, sigma_1: float, sigma_2: float, pi: float) -> None:
        super().__init__()
        mix = torch.distributions.Categorical(torch.tensor([pi, 1 - pi]))
        normals = torch.distributions.Normal(
            torch.zeros(2), torch.as_tensor([sigma_1, sigma_2])
        )
        self.distribution = torch.distributions.MixtureSameFamily(mix, normals)

    def log_prior(self, weight: Tensor) -> Tensor:
        return self.distribution.log_prob(weight).sum()
