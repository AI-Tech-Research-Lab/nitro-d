import numpy as np

from lib.layers.modules import Module
from lib.optim.Optimizer import Optimizer
from lib.utils.misc import truncated_division


class IntegerSGD(Optimizer):
    def __init__(self, layer: Module, weight_decay_inv: int = 0) -> None:
        super().__init__(layer, weight_decay_inv)
        assert layer.lr_amp_factor is not None
        self.lr_amp_factor = layer.lr_amp_factor

    def compute_updates(self, lr_inv: int, weight_gradient: np.ndarray,
                        bias_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        amplified_lr_inv = lr_inv * self.lr_amp_factor
        weight_update, bias_update = weight_gradient, bias_gradient

        # Compute the updates using the amplified learning rate
        weight_update = truncated_division(weight_update, amplified_lr_inv)
        if self.layer.bias is not None:
            bias_update = truncated_division(bias_update, amplified_lr_inv)

        if self.weight_decay > 0.0:
            # Add weight decay term without considering the learning rate
            wd_weights = truncated_division(self.layer.weights, self.weight_decay)
            weight_update = weight_update + wd_weights
            if self.layer.bias is not None:
                wd_bias = truncated_division(self.layer.bias, self.weight_decay)
                bias_update = bias_update + wd_bias

        # Return the updates
        return weight_update, bias_update
