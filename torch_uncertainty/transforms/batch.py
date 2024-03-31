import torch
from einops import rearrange
from torch import Tensor, nn


class RepeatTarget(nn.Module):
    def __init__(self, num_repeats: int) -> None:
        """Repeat the targets for ensemble training.

        Args:
            num_repeats: Number of times to repeat the targets.
        """
        super().__init__()

        if not isinstance(num_repeats, int):
            raise TypeError(
                f"num_repeats must be an integer. Got {num_repeats}."
            )
        if num_repeats <= 0:
            raise ValueError(
                f"num_repeats must be greater than 0. Got {num_repeats}."
            )

        self.num_repeats = num_repeats

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        return inputs, targets.repeat(
            self.num_repeats, *[1] * (targets.ndim - 1)
        )


class MIMOBatchFormat(nn.Module):
    def __init__(
        self, num_estimators: int, rho: float = 0.0, batch_repeat: int = 1
    ) -> None:
        """Format the batch for MIMO training.

        Args:
            num_estimators: Number of estimators.
            rho: Ratio of the correlation between the images for MIMO.
            batch_repeat: Number of times to repeat the batch.

        Reference:
            Havasi, M., et al. Training independent subnetworks for robust
            prediction. In ICLR, 2021.
        """
        super().__init__()

        if num_estimators <= 0:
            raise ValueError("num_estimators must be greater than 0.")
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho must be between 0 and 1.")
        if batch_repeat <= 0:
            raise ValueError("batch_repeat must be greater than 0.")

        self.num_estimators = num_estimators
        self.rho = rho
        self.batch_repeat = batch_repeat

    def shuffle(self, inputs: Tensor) -> Tensor:
        idx = torch.randperm(inputs.nelement(), device=inputs.device)
        return inputs.view(-1)[idx].view(inputs.size())

    def forward(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        inputs, targets = batch
        indexes = torch.arange(
            0, inputs.shape[0], device=inputs.device, dtype=torch.int64
        ).repeat(self.batch_repeat)
        main_shuffle = self.shuffle(indexes)
        threshold_shuffle = int(main_shuffle.shape[0] * (1.0 - self.rho))
        shuffle_indices = [
            torch.concat(
                [
                    self.shuffle(main_shuffle[:threshold_shuffle]),
                    main_shuffle[threshold_shuffle:],
                ],
                dim=0,
            )
            for _ in range(self.num_estimators)
        ]
        inputs = torch.stack(
            [
                torch.index_select(inputs, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            dim=0,
        )
        targets = torch.stack(
            [
                torch.index_select(targets, dim=0, index=indices)
                for indices in shuffle_indices
            ],
            dim=0,
        )
        inputs = rearrange(
            inputs, "m b c h w -> (m b) c h w", m=self.num_estimators
        )
        targets = rearrange(targets, "m b -> (m b)", m=self.num_estimators)
        return inputs, targets
