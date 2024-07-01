from typing import Literal

import cupy as cp
import numpy as np

from lib.layers.modules import Module
from lib.utils.misc import truncated_division


class PocketTanh(Module):
    """
    Element-wise hyperbolic tangent (tanh) function, rescaled in the range [-127, 127].
    It uses a piecewise linear approximation to work with integer-only arithmetics.
    """
    y_max, y_min = 127, -127
    int_max = np.iinfo(np.int32).max
    slopes_inv = [8, 2, 1, 2, 8, int_max, int_max]
    intervals = [y_max, 75, 32, -31, -74, y_min]

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.grad_inv = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cp.get_array_module(x)
        y = xp.full_like(x, self.y_max)
        conditions = [x < interval for interval in self.intervals]
        y[conditions[0]] = truncated_division(x[conditions[0]], 3) + 83
        y[conditions[1]] = x[conditions[1]] + 32
        y[conditions[2]] = x[conditions[2]] * 2
        y[conditions[3]] = x[conditions[3]] - 32
        y[conditions[4]] = truncated_division(x[conditions[4]], 3) - 83
        y[conditions[5]] = self.y_min

        if self.training:
            # Compute also the (inverse) gradient
            self.grad_inv = np.piecewise(x.astype(np.int32), conditions, self.slopes_inv)

        return y.astype(np.int8)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        return truncated_division(delta, self.grad_inv)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if 0 <= logging_level <= 1:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 2:
            return super().extra_repr(logging_level) + f'name={self.name})'


class PocketReLU(Module):
    """
    Element-wise rectified linear unit (ReLU) function, rescaled in the range [-127, 127].
    Unlike other pocket activation functions, no piecewise linear approximation is required.
    """
    y_max = 127

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.last_input = x.copy()
        return np.minimum(np.maximum(x, 0), self.y_max).astype(np.int8)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        return delta * np.logical_and(self.last_input > 0, self.last_input <= self.y_max)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if 0 <= logging_level <= 1:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 2:
            return super().extra_repr(logging_level) + f'name={self.name})'


class BipolarPocketReLU(PocketReLU):
    """
    Integer-compatible version of the Bipolar ReLU activation function. (Eidnes & Nøkland, 2017).
    It is identical to the original, but it uses the PocketReLU instead of the ReLU.
    """
    y_max = 127

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)
        self.mask = None

    def init_mask(self, x: np.ndarray) -> None:
        xp = cp.get_array_module(x)
        self.mask = xp.ones(x.shape, dtype=np.int8)
        if x.ndim == 2:
            self.mask[:, 1::2] = -1
        elif x.ndim == 4:
            self.mask[:, :, 0::2, 1::2] = -1
            self.mask[:, :, 1::2, 0::2] = -1
        else:
            raise ValueError("Input tensor must be 2D or 4D.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.mask is None:
            # Initialize the mask using the input shape
            self.init_mask(x)
        x_bipolar = x * self.mask
        y_bipolar = super().forward(x_bipolar)
        return y_bipolar * self.mask

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        delta_bipolar = delta * self.mask
        new_delta_bipolar = super().backward(delta_bipolar, lr_inv)
        return new_delta_bipolar * self.mask


class PocketLeakyReLU(Module):
    """
    Element-wise leaky rectified linear unit (LeakyReLU) function, rescaled in the range [-127, 127].
    It uses an integer division for the negative part of the input to work with integer-only arithmetic.

    Attributes
    ----------
    negative_slope_inv : int, default=4
        Inverse of the slope of the negative part of the function
    """
    y_max, y_min = 127, -127
    intervals = [y_max, 0, y_min]
    int_max = np.iinfo(np.int32).max
    supported_slope_inv = [3, 4, 5, 7, 8, 10, 15, 16, 20]

    def __init__(self, negative_slope_inv: int = 4, name: str = None) -> None:
        super().__init__(name=name)
        if negative_slope_inv not in self.supported_slope_inv:
            raise NotImplementedError(f"Unsupported negative_slope_inv: {negative_slope_inv}."
                                      f"Supported slopes are {self.supported_slope_inv}")
        self.negative_slope_inv = negative_slope_inv
        self.slopes_inv = [1, self.negative_slope_inv, self.int_max, self.int_max]
        self.last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.last_input = x.copy()
        positive_part = np.minimum(np.maximum(x, 0), self.y_max)
        negative_part = truncated_division(np.minimum(np.maximum(x, self.y_min), 0), self.negative_slope_inv)
        return (positive_part + negative_part).astype(np.int8)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        conditions = [self.last_input < interval for interval in self.intervals]
        grad_inv = np.piecewise(self.last_input.astype(np.int32), conditions, self.slopes_inv)
        return truncated_division(delta, grad_inv)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if logging_level == 0:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 1:
            return super().extra_repr(logging_level) + f'negative_slope_inv={self.negative_slope_inv})'
        if logging_level == 2:
            return super().extra_repr(
                logging_level) + f'negative_slope_inv={self.negative_slope_inv}, name={self.name})'


class NitroLeakyReLU(PocketLeakyReLU):
    """
    Mean-centered version of PocketLeakyReLU.
    It is still contained in the range [-127, 127].
    The mean correction term has been pre-computed.
    It uses an integer division for the negative part of the input to work with integer-only arithmetic.

    Attributes
    ----------
    negative_slope_inv : int, default=4
        Inverse of the slope of the negative part of the function
    """
    y_max, y_min = 127, -127
    mean_corrections = {3: 31, 4: 36, 5: 38, 7: 40, 8: 42, 10: 43, 15: 44, 16: 44, 20: 45}

    def __init__(self, negative_slope_inv: int = 4, name: str = None) -> None:
        super().__init__(negative_slope_inv=negative_slope_inv, name=name)
        self.mean_correction = self.mean_corrections[negative_slope_inv]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x) - self.mean_correction


class BipolarLeakyReLU(PocketLeakyReLU):
    """
    Integer-compatible version of the Bipolar LeakyReLU activation function. (Eidnes & Nøkland, 2017).
    It is identical to the original, but it uses the PocketLeakyReLU instead of the LeakyReLU.

    Attributes
    ----------
    negative_slope_inv : int, default=4
        Inverse of the slope of the negative part of the function
    """
    y_max, y_min = 127, -127

    def __init__(self, negative_slope_inv: int = 4, name: str = None) -> None:
        super().__init__(negative_slope_inv=negative_slope_inv, name=name)
        self.mask = None

    def init_mask(self, x: np.ndarray) -> None:
        xp = cp.get_array_module(x)
        self.mask = xp.ones(x.shape, dtype=np.int8)
        if x.ndim == 2:
            self.mask[:, 1::2] = -1
        elif x.ndim == 4:
            self.mask[:, :, 0::2, 1::2] = -1
            self.mask[:, :, 1::2, 0::2] = -1
        else:
            raise ValueError("Input tensor must be 2D or 4D.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.mask is None:
            # Initialize the mask using the input shape
            self.init_mask(x)
        x_bipolar = x * self.mask
        y_bipolar = super().forward(x_bipolar)
        return y_bipolar * self.mask

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        delta_bipolar = delta * self.mask
        new_delta_bipolar = super().backward(delta_bipolar, lr_inv)
        return new_delta_bipolar * self.mask
