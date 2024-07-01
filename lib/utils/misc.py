import cupy as cp
import numpy as np


def int_sqrt(n: int) -> int:
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def int_sqrt_vectorized(n: np.ndarray) -> np.ndarray:
    """ Vectorized version of int_sqrt which can also run on GPU """
    x = n.copy()
    y = (n + 1) // 2
    while np.any(y < x):
        mask = y < x
        x = y.copy()
        y[mask] = (x[mask] + n[mask] // x[mask]) // 2
    return x


def compute_num_elements(x: np.ndarray, axis: None | int | tuple[int, ...]) -> int:
    # Calculate the count of elements along the specified axis
    match axis:
        case None:
            return x.size
        case int():
            return x.shape[axis]
        case tuple():
            return np.prod([x.shape[a] for a in axis])


def int_mean(x: np.ndarray, axis: None | int | tuple[int, ...] = None, keepdims: bool = False) -> np.ndarray:
    x_sum = np.sum(x, axis=axis, keepdims=keepdims)
    return truncated_division(x_sum, compute_num_elements(x, axis))


def int_var(x: np.ndarray, axis: None | int | tuple[int, ...] = None, keepdims: bool = False) -> np.ndarray:
    mean, var = int_mean_and_var(x, axis=axis, keepdims=keepdims)
    return var


def int_mean_and_var(x: np.ndarray, axis: None | int | tuple[int, ...] = None,
                     keepdims: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mean = int_mean(x, axis=axis, keepdims=True)
    x_mean = (x - mean).astype(np.int64)
    var = int_mean(x_mean ** 2, axis=axis, keepdims=keepdims)
    if not keepdims:
        mean = mean.squeeze()
    return mean, var


def int_mad(x: np.ndarray, axis: None | int | tuple[int, ...] = None, keepdims: bool = True) -> np.ndarray:
    # Mean Absolute Deviation
    mean, mad = int_mean_and_mad(x, axis=axis, keepdims=keepdims)
    return mad


def int_mean_and_mad(x: np.ndarray, axis: None | int | tuple[int, ...] = None,
                     keepdims: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mean = int_mean(x, axis=axis, keepdims=keepdims)
    x_mean = x - mean
    mad = int_mean(np.abs(x_mean), axis=axis, keepdims=keepdims)
    return mean, mad


def truncated_division(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    """
    Perform truncated division, which removes the fractional part of the division result.
    This is different from the floored division, NumPy's default, which always rounds down.
    In this implementation, which supports vectorization, the numerator has to be an array
    but the denominator can also be a scalar.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator of the division
    denominator : np.ndarray | int
        Denominator of the division

    Returns
    -------
    np.ndarray
        The result of the truncated division
    """
    # Apply floored division
    result = numerator // denominator

    # Adjust the result for cases where numerator and denominator have different signs
    needs_adjustment = np.sign(numerator) != np.sign(denominator)
    has_remainder = numerator % denominator != 0
    adjustment = needs_adjustment & has_remainder
    result += adjustment

    # Return the result of the truncated division
    return result


def to_cpu(array: cp.ndarray):
    return cp.asnumpy(array, blocking=False)


def to_gpu(array: np.ndarray):
    return cp.asarray(array, blocking=False)
