from typing import Literal

import cupy as cp
import numpy as np

from lib.layers.modules import Module
from lib.utils.misc import truncated_division


class Flatten(Module):
    """
    Flatten layer which reshapes its input to a one-dimensional array.
    Used to connect convolutional layers to linear layers.

    Attributes
    ----------
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.last_input_shape = None

    def forward(self, x: np.ndarray):
        if self.training:
            self.last_input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        return delta.reshape(self.last_input_shape)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if 0 <= logging_level <= 1:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 2:
            return super().extra_repr(logging_level) + f'name={self.name})'


class ConstantScaling(Module):
    """
    Constant scaling layer which applies a scaling factor to shift the values to an appropriate order of magnitude.
    Without it, values will all end up in the saturated parts of the activation functions.
    The scaling factors should be computed on the basis of the values' bit-width.
    Scaling during the forward pass is mandatory, scaling during the backward pass is optional.

    Attributes
    ----------
    fwd_factor : int
        Scaling factor to be applied in the forward pass.
    bwd_factor : int, default=1
        Scaling factor to be applied in the backward pass.
    """

    def __init__(self, fwd_factor: int, bwd_factor: int = 1, name: str = None) -> None:
        super().__init__(name=name)
        self.fwd_factor = fwd_factor
        self.bwd_factor = bwd_factor

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = truncated_division(x, self.fwd_factor)
        return np.clip(y, -127, 127).astype(np.int8)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        return delta

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if logging_level == 0:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 1:
            return super().extra_repr(logging_level) + f'fwd_factor={self.fwd_factor}, bwd_factor={self.bwd_factor})'
        if logging_level == 2:
            return super().extra_repr(logging_level) + (f'fwd_factor={self.fwd_factor}, '
                                                        f'bwd_factor={self.bwd_factor}, name={self.name})')


class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor with a given probability.
    The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.
    It is required to rescale the input so that the average magnitude is preserved.
    During evaluation, the module simply returns the input.

    Attributes
    ----------
    dropout_rate : float, default=0.5
        The probability that each element is set to zero.
    """

    # Mapping between dropout rate and (1000 * rescaling factor)
    rescaling_factors = {0.05: 1052, 0.10: 1111, 0.15: 1176, 0.20: 1250, 0.25: 1333, 0.30: 1428, 0.35: 1538,
                         0.40: 1666, 0.45: 1818, 0.50: 2000, 0.55: 2222, 0.60: 2500, 0.65: 2857, 0.70: 3333,
                         0.75: 4000, 0.8: 5000, 0.85: 6666, 0.9: 10000, 0.95: 20000}

    def __init__(self, dropout_rate: float = 0.5, name: str = None) -> None:
        if dropout_rate not in self.rescaling_factors.keys():
            raise ValueError('Unsupported dropout rate. Supported values are multiples of 0.05 from 0.05 to 0.95')

        super().__init__(name=name)
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            # Create a binary mask using the dropout rate
            xp = cp.get_array_module(x)
            self.mask = xp.random.binomial(1, p=(1 - self.dropout_rate), size=x.shape).astype(np.bool_)

            # Apply the mask to the input
            masked_input = x * self.mask
            masked_input = masked_input.astype(np.int32) * self.rescaling_factors[self.dropout_rate]
            return truncated_division(masked_input, 1000).astype(np.int16)
        else:
            return x

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        masked_delta = delta * self.mask
        masked_delta = masked_delta.astype(np.int32) * self.rescaling_factors[self.dropout_rate]
        return truncated_division(masked_delta, 1000)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if logging_level == 0:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 1:
            return super().extra_repr(logging_level) + f'dropout_rate={self.dropout_rate})'
        if logging_level == 2:
            return super().extra_repr(logging_level) + (f'dropout_rate={self.dropout_rate}, '
                                                        f'rescaling_factor={self.rescaling_factors[self.dropout_rate]},'
                                                        f' name={self.name})')
