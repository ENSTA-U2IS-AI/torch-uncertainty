# fmt: off
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init

from .sampler import PriorDistribution, TrainableDistribution


# fmt: on
class BayesLinear(nn.Module):
    """Bayesian Linear Layer with Mixture of Normals prior and Normal posterior

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        prior_mu (float, optional): Mean of the prior distribution
            Defaults to 0.0.
        prior_sigma_1 (float, optional): Standard deviation of the first prior
            distribution. Defaults to 0.1.
        prior_sigma_2 (float, optional): Standard deviation of the second prior
            distribution. Defaults to 0.1.
        prior_pi (float, optional): Mixture control variable. Defaults to 0.1.
        mu_init (float, optional): Initial mean of the posterior distribution.
            Defaults to 0.0.
        sigma_init (float, optional): Initial standard deviation of the
            posterior distribution. Defaults to -7.0.
        frozen (bool, optional): Whether to freeze the posterior distribution.
            Defaults to False.
        bias (bool, optional): Whether to use a bias term. Defaults to True.
        device (optional): Device to use. Defaults to None.
        dtype (optional): Data type to use. Defaults to None.

    Paper Reference:
        Blundell, Charles, et al. "Weight uncertainty in neural networks"
            ICML 2015.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.4,
        prior_pi: float = 1,
        mu_init: float = 0.0,
        sigma_init: float = -7.0,
        frozen: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.frozen = frozen
        self.bias = bias

        self.weight_mu = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if self.bias:
            self.bias_mu = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_log_sigma", None)

        self.reset_parameters()
        self.weight_sampler = TrainableDistribution(
            self.weight_mu, self.weight_sigma
        )
        if self.bias:
            self.bias_sampler = TrainableDistribution(
                self.bias_mu, self.bias_sigma
            )

        self.weight_prior_dist = PriorDistribution(
            prior_sigma_1, prior_sigma_2, prior_pi
        )
        if self.bias:
            self.bias_prior_dist = PriorDistribution(
                prior_sigma_1, prior_sigma_2, prior_pi
            )
        self.lprior = 0
        self.lvposterior = 0

    def reset_parameters(self) -> None:
        # TODO: change init
        init.normal_(self.weight_mu, mean=self.mu_init, std=0.1)
        init.normal_(self.weight_sigma, mean=self.sigma_init, std=0.1)

        if self.bias:
            init.normal_(self.bias_mu, mean=self.mu_init, std=0.1)
            init.normal_(self.bias_sigma, mean=self.sigma_init, std=0.1)

    def forward(self, input: Tensor) -> Tensor:
        if self.frozen:
            return self._frozen_forward(input)
        else:
            return self._forward(input)

    def _frozen_forward(self, input):
        return F.linear(
            input, self.weight_mu, self.bias_mu if self.bias else None
        )

    def _forward(self, input: Tensor) -> Tensor:
        weight = self.weight_sampler.sample()

        if self.bias:
            bias = self.bias_sampler.sample()
            bias_lposterior = self.bias_sampler.log_posterior()
            bias_lprior = self.bias_prior_dist.log_prior(bias)
        else:
            bias, bias_lposterior, bias_lprior = None, 0, 0

        self.lvposterior = self.weight_sampler.log_posterior() + bias_lposterior
        self.lprior = self.weight_prior_dist.log_prior(weight) + bias_lprior

        return F.linear(input, weight, bias)

    def freeze(self) -> None:
        """Freeze the layer by setting the frozen attribute to True."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the layer by setting the frozen attribute to False."""
        self.frozen = False

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
