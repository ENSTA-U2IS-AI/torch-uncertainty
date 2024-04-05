from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class PolyLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 0.9,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ) -> None:
        self.power = power
        self.total_iters = total_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self) -> list[float]:
        return [
            max(
                base_lr
                * (1 - self.last_epoch / self.total_iters) ** self.power,
                self.min_lr,
            )
            for base_lr in self.base_lrs
        ]
