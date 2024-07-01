from typing import Literal

import numpy as np

from lib.utils.enums import Initialization
from lib.utils.misc import int_sqrt


def init_params(xp, init_type, weight_shape, bias_shape, in_features, dtype):
    match init_type:
        case Initialization.UNIFORM_KAIMING_LEAKY_RELU:
            bound = integer_kaiming_uniform(weight_shape, mode='fan_in', non_linearity='leaky_relu',
                                            negative_slope_inv=4)

        case Initialization.UNIFORM_KAIMING_RELU:
            bound = integer_kaiming_uniform(weight_shape, mode='fan_in', non_linearity='relu')

        case Initialization.UNIFORM_XAVIER:
            bound = integer_xavier_uniform(weight_shape, non_linearity='tanh')

        case Initialization.UNIFORM_STD:
            dtype_max = np.iinfo(np.int16).max
            bound = int_sqrt((64 * dtype_max) // in_features)

        case Initialization.UNIFORM_FIXED:
            bound = 64

        case Initialization.ZEROS:
            weights = xp.zeros(weight_shape, dtype=dtype)
            bias = xp.zeros(bias_shape, dtype=dtype) if bias_shape else None
            return weights, bias

        case Initialization.ONES:
            weights = xp.ones(weight_shape, dtype=dtype)
            bias = xp.ones(bias_shape, dtype=dtype) if bias_shape else None
            return weights, bias

        case _:
            raise ValueError(f"Unsupported initialization type {init_type}")

    bound = np.minimum(bound, 64)
    weights = xp.random.randint(-bound, bound + 1, size=weight_shape, dtype=dtype)
    bias = xp.random.randint(-bound, bound + 1, size=bias_shape, dtype=dtype) if bias_shape else None
    return weights, bias


def integer_kaiming_uniform(shape: tuple[int, ...], mode: Literal['fan_in', 'fan_out'] = 'fan_in',
                            non_linearity: str = 'leaky_relu', negative_slope_inv: int = 5) -> int:
    """
    Fills the input `Tensor` with values according to the method described in `Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification` - He, K. et al. (2015), using a uniform
    distribution.
    Also known as He initialization.

    Parameters
    ----------
    shape: tuple[int, ...]
        The shape of the tensor to initialize
    mode: Literal['fan_in', 'fan_out'], default='fan_in'
        The mode to use for computing the fan
    non_linearity: str, default='leaky_relu'
        The non-linear function
    negative_slope_inv: int, default=5
        The inverse negative slope of the non-linear function used in the non-linearity

    Returns
    -------
    int
        The recommended bound for the uniform distribution
    """
    fan = _calculate_fan(shape, mode)
    gain = _calculate_integer_gain(non_linearity, negative_slope_inv)
    int_range = 128
    # Perform the divisions last to minimize the truncation error
    bound = (gain * int_range * 1732) // (int_sqrt(fan) * 1000 * 1000)
    return bound


def integer_xavier_uniform(shape: tuple[int, ...], non_linearity: str = 'tanh') -> int:
    """
    Fill the input Tensor with values using a Xavier uniform distribution.
    The method is described in `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
    Also known as Glorot initialization.

      Parameters
    ----------
    shape: tuple[int, ...]
        The shape of the tensor to initialize
    non_linearity: str, default='tanh'
        The non-linear function

    Returns
    -------
    int
        The recommended bound for the uniform distribution
    """
    fan_in = _calculate_fan(shape, 'fan_in')
    fan_out = _calculate_fan(shape, 'fan_out')
    gain = _calculate_integer_gain(non_linearity)
    int_range = 128
    # Perform the divisions last to minimize the truncation error
    bound = (gain * int_range * 2449) // (int_sqrt(fan_in + fan_out) * 1000 * 1000)
    return bound


def _calculate_fan(shape: tuple[int, ...], mode: str = 'fan_in') -> int:
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_maps = shape[1]
    num_output_maps = shape[0]
    receptive_field_size = 1

    if len(shape) > 2:
        for s in shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_maps * receptive_field_size
    fan_out = num_output_maps * receptive_field_size

    return fan_in, fan_out


def _calculate_integer_gain(non_linearity: str, negative_slope_inv: int = 5) -> int:
    """
    Return the recommended gain value for the given non-linearity function.
    The values are as follows

    Parameters
    ----------
    non_linearity: str
        The non-linear function (`nn.functional` name)
    negative_slope_inv: int, default=5
        Optional parameter for the non-linear function

    Returns
    -------
    The value of the gain multiplied by 1000
    """
    match non_linearity.lower():
        case 'tanh':
            # Return 5 / 3 * 1000
            return 1667

        case 'relu':
            # Return sqrt(2) * 1000
            return 1414

        case 'leaky_relu':
            gains = {3: 1342, 4: 1372, 5: 1387, 7: 1400, 8: 1403, 10: 1407, 15: 1411, 16: 1411, 20: 1412}
            if negative_slope_inv not in gains.keys():
                raise NotImplementedError(f"Unsupported negative_slope_inv {negative_slope_inv}")
            # Return sqrt(2 / (1 + negative_slope^2)) * 1000
            return gains[negative_slope_inv]

        case _:
            raise ValueError(f"Unsupported non_linearity {non_linearity}")
