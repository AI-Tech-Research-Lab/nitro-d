import numpy as np

from lib.optim.LRScheduler import LRScheduler


class ConstantLR(LRScheduler):
    def _compute_lr(self) -> int:
        return self.lr


class LinearLR(LRScheduler):
    """
    Decay the learning rate by a fixed percentage of the base learning rate at each epoch,
    until a maximum learning rate is reached.

    Attributes
    _________
    base_lr: int
        Base learning rate
    factor: int
        Percentage of the base learning rate to be reduced at each epoch
    maximum_lr: int
        Maximum learning rate
    """

    def __init__(self, base_lr: int, factor: int, maximum_lr: int, from_epoch: int = 0, verbose: bool = False) -> None:
        if factor < 0 or factor > 100:
            raise ValueError(f'Invalid factor {factor}. Must be in the range [0, 100].')
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.factor = factor
        self.maximum_lr = maximum_lr

    def _compute_lr(self) -> int:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            return np.minimum(self.lr + (self.base_lr * self.factor // 100), self.maximum_lr)


class ExponentialLR(LRScheduler):
    """
    Decay the learning rate by a factor of the current learning rate at each epoch,
    until a maximum learning rate is reached.

    Attributes
    _________
    base_lr: int
        Base learning rate
    factor: int
        Percentage of the current learning rate to be reduced at each epoch
    maximum_lr: int
        Maximum learning rate
    """

    def __init__(self, base_lr: int, factor: int, maximum_lr: int, from_epoch: int = 0, verbose: bool = False) -> None:
        if factor < 0 or factor > 100:
            raise ValueError(f'Invalid factor {factor}. Must be in the range [0, 100].')
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.factor = factor
        self.maximum_lr = maximum_lr

    def _compute_lr(self) -> int:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            return np.minimum(self.lr * (100 + self.factor) // 100, self.maximum_lr)


class HalvingLR(LRScheduler):
    def __init__(self, base_lr: int, num_epochs: int, maximum_lr: int,
                 from_epoch: int = 0, verbose: bool = False) -> None:
        super().__init__(base_lr, from_epoch=from_epoch, verbose=verbose)
        self.num_epochs = num_epochs
        self.maximum_lr = maximum_lr

    def _compute_lr(self) -> int:
        if self.last_epoch <= self.from_epoch:
            return self.base_lr
        else:
            if (self.last_epoch - self.from_epoch) % self.num_epochs == 0:
                return np.minimum(self.lr * 2, self.maximum_lr)
            else:
                return self.lr
