from collections.abc import Callable

import torch
from torch.optim.optimizer import Optimizer, ParamsT


class SGLD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        dataset_size: int,
        lr: float = 1e-5,
        weight_decay: float = 0,
    ) -> None:
        r"""Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

        TODO: dynamical equations

        Args:
            params (ParamsT): Iterable of parameters or named_parameters to optimize or iterable of
                dicts defining parameter groups. When using named_parameters, all parameters in all
                groups should be named.
            dataset_size (int): Size of the dataset. Used to scale the gradient.
            lr (float): Learning rate. Defaults to ``1e-3``.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to ``0``.

        Reference:
            - `Bayesian Learning via Stochastic Gradient Langevin Dynamics <https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf>`_.
        """
        defaults = {"dataset_size": dataset_size, "lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            dataset_size = group["dataset_size"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay / dataset_size)

                noise = torch.randn_like(p.data)
                p.data.add_(-lr * dataset_size * 0.5 * d_p - ((lr**0.5) * noise))

        return loss
