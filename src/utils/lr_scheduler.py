"""Learning-rate schedules indexed by completed optimizer updates."""

from math import cos, pi

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """Linearly warm up to optimizer LR, then decay to eta_min by T_max."""

    def __init__(
        self,
        optimizer,
        T_max,
        warmup_steps=1000,
        warmup_start_lr=3e-6,
        eta_min=3e-6,
        last_epoch=-1,
    ):
        if not 0 <= warmup_steps < T_max:
            raise ValueError("Require 0 <= warmup_steps < T_max.")
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(max(self.last_epoch, 0), self.T_max)
        if step < self.warmup_steps:
            fraction = step / self.warmup_steps
            return [
                self.warmup_start_lr + (peak - self.warmup_start_lr) * fraction
                for peak in self.base_lrs
            ]
        fraction = (step - self.warmup_steps) / (self.T_max - self.warmup_steps)
        return [
            self.eta_min + (peak - self.eta_min) * (1 + cos(pi * fraction)) / 2
            for peak in self.base_lrs
        ]
