from abc import ABC, abstractmethod


class LRScheduler(ABC):
    def __init__(self, base_lr: int, from_epoch: int = 0, verbose: bool = False) -> None:
        self.last_epoch: int = -1
        self.base_lr = base_lr
        self.from_epoch = from_epoch
        self.verbose = verbose
        self.lr = base_lr

    def get_lr(self) -> int:
        return self.lr

    def set_lr(self, lr: int) -> None:
        self.lr = lr

    @abstractmethod
    def _compute_lr(self) -> int:
        """ Compute learning rate using the scheduler """
        pass

    def print_lr(self) -> None:
        print(f'Epoch {self.last_epoch:>2}: adjusting learning rate to {self.lr}')

    def step(self) -> None:
        self.last_epoch += 1
        old_lr = self.lr
        self.lr = self._compute_lr()

        # Log the learning rate only when it changes
        if self.verbose and old_lr != self.lr:
            self.print_lr()

