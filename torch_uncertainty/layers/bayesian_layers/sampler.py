# fmt: off
import torch
import torch.nn as nn

import numpy as np


# fmt: on
class TrainableDistribution(nn.Module):
    lsqrt2pi = np.log(np.sqrt(2 * np.pi))

    def __init__(self, mu, rho):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer("eps_w", torch.Tensor(self.mu.shape))
        self.sigma = None
        self.weight = None

    def sample(self):
        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.weight = self.mu + self.sigma * self.eps_w
        return self.weight

    def log_posterior(self, weight=None):
        assert (
            self.weight is not None
        ), "Sample the weights before asking for the log posterior."
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
    def __init__(self, sigma: float):
        super().__init__()
        self.distribution = torch.distributions.Normal(0, sigma)

    def log_prior(self, w):
        log_prob = self.distribution.log_prob(w)
        return (log_prob - 0.5).sum()
