import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .swa import SWA


def flatten(lst: list[Tensor]) -> Tensor:
    tmp = [i.view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, like_tensor_list):
    """Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    shaped like like_tensor_list.

    """
    out_list = []
    i = 0
    for tensor in like_tensor_list:
        n = tensor.numel()
        out_list.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return out_list


class SWAG(SWA):
    def __init__(
        self,
        model: nn.Module,
        cycle_start: int,
        cycle_length: int,
        scale: float = 1.0,
        diag_covariance: bool = True,
        max_num_models: int = 20,
        var_clamp: float = 1e-30,
        num_estimators: int = 16,
    ) -> None:
        """Stochastic Weight Averaging Gaussian (SWAG).

        Args:
            model (nn.Module): PyTorch model to be trained.
            cycle_start (int): Epoch to start SWAG.
            cycle_length (int): Number of epochs between SWAG updates.
            scale (float, optional): Scale of the Gaussian. Defaults to 1.0.
            diag_covariance (bool, optional): Whether to use a diagonal covariance. Defaults to False.
            max_num_models (int, optional): Maximum number of models to store. Defaults to 0.
            var_clamp (float, optional): Minimum variance. Defaults to 1e-30.
            num_estimators (int, optional): Number of posterior estimates to use. Defaults to 16.

        Reference:
            Maddox, W. J. et al. A simple baseline for bayesian uncertainty in
            deep learning. In NeurIPS 2019.

        Note:
            Modified from https://github.com/wjmaddox/swa_gaussian
        """
        super().__init__(model, cycle_start, cycle_length)
        _swag_checks(scale, max_num_models, var_clamp)

        self.num_models = 0
        self.num_estimators = num_estimators
        self.scale = scale

        self.diag_covariance = diag_covariance
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

        self.swag_params = []
        self.swag_model = copy.deepcopy(model)
        self.swag_model.apply(lambda module: self.extract_parameters(module))

        self.fit = False
        self.samples = []

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fit:
            return self.model.forward(x)
        return torch.cat([mod(x) for mod in self.samples])

    def extract_parameters(self, module: nn.Module) -> None:
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            data = module._parameters[name].data
            delattr(module, name)

            mean, squared_mean = torch.zeros_like(data), torch.zeros_like(data)
            module.register_buffer(f"{name}_mean", mean)
            module.register_buffer(f"{name}_sq_mean", squared_mean)

            if not self.diag_covariance:
                covariance_sqrt = torch.zeros((0, data.numel()))
                module.register_buffer(
                    f"{name}_covariance_sqrt", covariance_sqrt
                )

            self.swag_params.append((module, name))

    @torch.no_grad()
    def update_model(self, epoch: int) -> None:
        if (
            epoch >= self.cycle_start
            and (epoch - self.cycle_start) % self.cycle_length == 0
        ):
            print("update SWAG model")
            for (module, name), param in zip(
                self.swag_params, self.model.parameters(), strict=False
            ):
                mean = module.__getattr__(f"{name}_mean")
                squared_mean = module.__getattr__(f"{name}_sq_mean")
                new_param = param.data

                mean = mean * self.num_models / (
                    self.num_models + 1
                ) + new_param / (self.num_models + 1)
                squared_mean = squared_mean * self.num_models / (
                    self.num_models + 1
                ) + new_param**2 / (self.num_models + 1)

                module.__setattr__(f"{name}_mean", mean)
                module.__setattr__(f"{name}_sq_mean", squared_mean)

                if not self.diag_covariance:
                    covariance_sqrt = module.__getattr__(
                        f"{name}_covariance_sqrt"
                    )
                    dev = (new_param - mean).view(-1, 1).t()
                    covariance_sqrt = torch.cat((covariance_sqrt, dev), dim=0)
                    if self.num_models + 1 > self.max_num_models:
                        covariance_sqrt = covariance_sqrt[1:, :]
                    module.__setattr__(
                        f"{name}_covariance_sqrt", covariance_sqrt
                    )

            self.num_models += 1

            self.samples = []
            for _ in range(self.num_estimators):
                self.sample(self.scale, self.diag_covariance)
                self.samples.append(copy.deepcopy(self.swag_model))
            self.need_bn_update = True
            self.fit = True

    def update_bn(self, loader: DataLoader, device) -> None:
        if self.need_bn_update:
            for mod in self.samples:
                torch.optim.swa_utils.update_bn(loader, mod, device=device)
            self.need_bn_update = False

    def sample(
        self,
        scale: float,
        diag_covariance: bool | None = None,
        block: bool = False,
        seed: int | None = None,
    ) -> None:  # TODO: Fix sampling
        if seed is not None:
            torch.manual_seed(seed)

        if diag_covariance is None:
            diag_covariance = self.diag_covariance
        if not diag_covariance and self.diag_covariance:
            raise ValueError(
                "Cannot sample full rank from diagonal covariance matrices"
            )

        if not block:
            self._fullrank_sample(scale, diag_covariance)
        else:
            raise NotImplementedError("Raise an issue if you need this feature")

    def _fullrank_sample(self, scale: float, diagonal_covariance: bool) -> None:
        mean_list, sq_mean_list = [], []
        if not diagonal_covariance:
            cov_mat_sqrt_list = []

        for module, name in self.swag_params:
            mean = module.__getattr__(f"{name}_mean")
            sq_mean = module.__getattr__(f"{name}_sq_mean")

            if not diagonal_covariance:
                cov_mat_sqrt = module.__getattr__(f"{name}_covariance_sqrt")
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if not diagonal_covariance:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            # vérifier le min
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale**0.5 * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(
            self.swag_params, samples_list, strict=False
        ):
            module.__setattr__(name, sample.cuda())

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        super().load_state_dict(state_dict, strict)

    def compute_logdet(self, block=False):
        raise NotImplementedError("Raise an issue if you need this feature")

    def compute_logprob(self, vec=None, block=False, diag=False):
        raise NotImplementedError("Raise an issue if you need this feature")


def _swag_checks(scale: float, max_num_models: int, var_clamp: float) -> None:
    if scale < 0:
        raise ValueError(f"`scale` must be non-negative. Got {scale}.")
    if max_num_models < 0:
        raise ValueError(
            f"`max_num_models` must be non-negative. Got {max_num_models}."
        )
    if var_clamp < 0:
        raise ValueError(f"`var_clamp` must be non-negative. Got {var_clamp}.")
