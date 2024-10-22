from importlib import util

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

if util.find_spec("scipy"):
    import scipy

    scipy_installed = True
else:  # coverage: ignore
    scipy_installed = False


def beta_warping(x, alpha_cdf: float = 1.0, eps: float = 1e-12) -> float:
    return scipy.stats.beta.cdf(x, a=alpha_cdf + eps, b=alpha_cdf + eps)


def sim_gauss_kernel(dist, tau_max: float = 1.0, tau_std: float = 0.5) -> float:
    dist_rate = tau_max * np.exp(
        -(dist - 1) / (np.mean(dist) * 2 * tau_std * tau_std)
    )
    return 1 / (dist_rate + 1e-12)


# ruff: noqa: ERA001
# def tensor_linspace(start: Tensor, stop: Tensor, num: int):
#     """
#     Creates a tensor of shape [num, *start.shape] whose values are evenly
#     spaced from start to end, inclusive.
#     Replicates but the multi-dimensional behaviour of numpy.linspace in PyTorch.
#     """
#     # create a tensor of 'num' steps from 0 to 1
#     steps = torch.arange(num, dtype=torch.float32, device=start.device) / (
#         num - 1
#     )

#     # reshape the 'steps' tensor to [-1, *([1]*start.ndim)]
#     # to allow for broadcastings
#     # using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here
#     # but torchscript
#     # "cannot statically infer the expected size of a list in this context",
#     # hence the code below
#     for i in range(start.ndim):
#         steps = steps.unsqueeze(-1)

#     # the output starts at 'start' and increments until 'stop' in each dimension
#     out = start[None] + steps * (stop - start)[None]

#     return out


# def torch_beta_cdf(
#     x: Tensor, c1: Tensor | float, c2: Tensor | float, npts=100, eps=1e-12
# ):
#     if isinstance(c1, float):
#         if c1 == c2:
#             c1 = Tensor([c1], device=x.device)
#             c2 = c1
#         else:
#             c1 = Tensor([c1], device=x.device)
#     if isinstance(c2, float):
#         c2 = Tensor([c2], device=x.device)
#     bt = torch.distributions.Beta(c1, c2)

#     if isinstance(x, float):
#         x = Tensor(x)

#     X = tensor_linspace(torch.zeros_like(x) + eps, x, npts)
#     return torch.trapezoid(bt.log_prob(X).exp(), X, dim=0)


# def torch_beta_warping(
#     x: Tensor, alpha_cdf: float | Tensor = 1.0, eps=1e-12, npts=100
# ):
#     return torch_beta_cdf(
#         x=x, c1=alpha_cdf + eps, c2=alpha_cdf + eps, npts=npts, eps=eps
#     )


# def torch_sim_gauss_kernel(dist: Tensor, tau_max=1.0, tau_std=0.5):
#     dist_rate = tau_max * torch.exp(
#         -(dist - 1) / (torch.mean(dist) * 2 * tau_std * tau_std)
#     )

#     return 1 / (dist_rate + 1e-12)


# TODO: Should be a torchvision transform
class AbstractMixup(nn.Module):
    def __init__(
        self, alpha: float = 1.0, mode: str = "batch", num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        self.alpha = alpha
        self.num_classes = num_classes
        self.mode = mode

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = self.rng.beta(a=self.alpha, b=self.alpha)
        else:
            lam = torch.as_tensor(
                self.rng.beta(a=self.alpha, b=self.alpha, size=batch_size),
                device=device,
            )
        index = torch.randperm(batch_size, device=device)
        return lam, index

    def _linear_mixing(
        self,
        lam: Tensor | float,
        inp: Tensor,
        index: Tensor,
    ) -> Tensor:
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(inp.ndim - 1)]).float()
        return lam * inp + (1 - lam) * inp[index, :]

    def _mix_target(
        self,
        lam: Tensor | float,
        target: Tensor,
        index: Tensor,
    ) -> Tensor:
        y1 = F.one_hot(target, self.num_classes)
        y2 = F.one_hot(target[index], self.num_classes)
        if isinstance(lam, Tensor):
            lam = lam.view(-1, *[1 for _ in range(y1.ndim - 1)]).float()

        return lam * y1 + (1 - lam) * y2

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class Mixup(AbstractMixup):
    """Original Mixup method from Zhang et al.

    Reference:
        "mixup: Beyond Empirical Risk Minimization" (ICLR 2021)
        http://arxiv.org/abs/1710.09412.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        mixed_x = self._linear_mixing(lam, x, index)
        mixed_y = self._mix_target(lam, y, index)
        return mixed_x, mixed_y


class MixupIO(AbstractMixup):
    """Mixup on inputs only with targets unchanged, from Wang et al.

    Reference:
        "On the Pitfall of Mixup for Uncertainty Calibration" (CVPR 2023)
        https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        mixed_x = self._linear_mixing(lam, x, index)

        if self.mode == "batch":
            mixed_y = self._mix_target(float(lam > 0.5), y, index)
        else:
            mixed_y = self._mix_target((lam > 0.5).float(), y, index)

        return mixed_x, mixed_y


class RegMixup(AbstractMixup):
    """RegMixup method from Pinto et al.

    Reference:
        'RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness' (NeurIPS 2022)
        https://arxiv.org/abs/2206.14502.
    """

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)
        part_x = self._linear_mixing(lam, x, index)
        part_y = self._mix_target(lam, y, index)
        mixed_x = torch.cat([x, part_x], dim=0)
        mixed_y = torch.cat([F.one_hot(y, self.num_classes), part_y], dim=0)
        return mixed_x, mixed_y


class WarpingMixup(AbstractMixup):
    def __init__(
        self,
        alpha: float = 1.0,
        mode: str = "batch",
        num_classes: int = 1000,
        apply_kernel: bool = True,
        tau_max: float = 1.0,
        tau_std: float = 0.5,
    ) -> None:
        """Kernel Warping Mixup method from Bouniot et al.

        Reference:
            "Tailoring Mixup to Data using Kernel Warping functions" (2023)
            https://arxiv.org/abs/2311.01434.
        """
        if not scipy_installed:  # coverage: ignore
            raise ImportError(
                "The scipy library is not installed. Please install"
                "torch_uncertainty with the all option:"
                """pip install -U "torch_uncertainty[all]"."""
            )
        super().__init__(alpha=alpha, mode=mode, num_classes=num_classes)
        self.apply_kernel = apply_kernel
        self.tau_max = tau_max
        self.tau_std = tau_std

    def _get_params(self, batch_size: int, device: torch.device):
        if self.mode == "batch":
            lam = self.rng.beta(a=self.alpha, b=self.alpha)
        else:
            lam = self.rng.beta(a=self.alpha, b=self.alpha, size=batch_size)

        index = torch.randperm(batch_size, device=device)
        return lam, index

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        feats: Tensor,
        warp_param: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        lam, index = self._get_params(x.size()[0], x.device)

        if self.apply_kernel:
            l2_dist = (
                (feats - feats[index])
                .pow(2)
                .sum([i for i in range(len(feats.size())) if i > 0])
                .cpu()
                .numpy()
            )
            warp_param = sim_gauss_kernel(l2_dist, self.tau_max, self.tau_std)

        k_lam = torch.as_tensor(beta_warping(lam, warp_param), device=x.device)
        mixed_x = self._linear_mixing(k_lam, x, index)
        mixed_y = self._mix_target(k_lam, y, index)
        return mixed_x, mixed_y
