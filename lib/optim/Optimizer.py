from abc import ABC, abstractmethod

import numpy as np

from lib.layers.modules import Module


class Optimizer(ABC):
    def __init__(self, layer: Module, weight_decay: int = 0, shift_when_possible: bool = False) -> None:
        self.layer = layer
        self.weight_decay = weight_decay
        self.shift_when_possible = shift_when_possible

    @abstractmethod
    def compute_updates(self, lr_inv: int, weight_gradient: np.ndarray,
                        bias_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass
