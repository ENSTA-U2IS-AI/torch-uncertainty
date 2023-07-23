# fmt: off
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np


# fmt: on
class TrainableDistribution(nn.Module):
    lsqrt2pi = torch.as_tensor(np.log(np.sqrt(2 * np.pi)))

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
        w_sample = torch.normal(
            mean=0, std=1, size=self.mu.shape, device=self.mu.device
        )
        self.sigma = torch.log1p(torch.exp(self.rho)).to(self.mu.device)
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
    def __init__(
        self,
        sigma_1: float,
        sigma_2: float,
        pi: float,
    ) -> None:
        super().__init__()
        self.pi = torch.tensor([pi, 1 - pi])
        self.mus = torch.zeros(2)
        self.sigmas = torch.as_tensor([sigma_1, sigma_2])

    def log_prior(self, weight: Tensor) -> Tensor:
        self.convert(weight.device)
        mix = torch.distributions.Categorical(self.pi)
        normals = torch.distributions.Normal(self.mus, self.sigmas)
        self.distribution = torch.distributions.MixtureSameFamily(mix, normals)
        return self.distribution.log_prob(weight).sum()

    def convert(self, device) -> None:
        self.pi = self.pi.to(device)
        self.mus = self.mus.to(device)
        self.sigmas = self.sigmas.to(device)
