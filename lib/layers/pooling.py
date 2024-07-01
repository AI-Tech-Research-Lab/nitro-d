from typing import Literal

import cupy as cp
import numpy as np

from lib.layers.modules import Module
from lib.utils.misc import int_mean, truncated_division


def add_inf_padding(x: np.ndarray, pad_sizes: tuple[int, int], dtype: str = 'int32') -> np.ndarray:
    """
    Apply negative infinite padding to the last two dimensions of a 4D array.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    pad_sizes: tuple[int, int]
        Number of rows and columns to pad the input with
    dtype: str, default='int32'
        Data type of the padded array.
        It is used to determine the minimum value for negative infinite padding

    Returns
    -------
    x_pad: np.ndarray
        Padded array of shape (batch_size, num_channels, height + 2 * pad_sizes[0], width + 2 * pad_sizes[1])
    """
    pad_width = np.array([(0, 0), (0, 0), (pad_sizes[0], pad_sizes[0]), (pad_sizes[1], pad_sizes[1])])
    constant_values = np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else -np.inf
    return np.pad(x, pad_width=pad_width, constant_values=constant_values)


def add_zero_padding(x: np.ndarray, pad_sizes: tuple[int, int]) -> np.ndarray:
    """
    Apply zero padding to the last two dimensions of a 4D array.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    pad_sizes: tuple[int, int]
        Number of rows and columns to pad the input with

    Returns
    -------
    x_pad: np.ndarray
        Padded array of shape (batch_size, num_channels, height + 2 * pad_sizes[0], width + 2 * pad_sizes[1])
    """
    pad_width = np.array([(0, 0), (0, 0), (pad_sizes[0], pad_sizes[0]), (pad_sizes[1], pad_sizes[1])])
    return np.pad(x, pad_width=pad_width, constant_values=0)


def remove_padding(delta: np.ndarray, x_shape: tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract the original input errors from the padded ones.
    Used in the backward pass.

    Parameters
    ----------
    delta: np.ndarray
        The 4D array of errors to be extracted
    x_shape: tuple[int, int, int, int]
        The shape of the original input to the layer

    Returns
    -------
    delta: np.ndarray
        The extracted errors
    """
    m, Nc, Nhd, Nwd = delta.shape
    m, Nc, Nhx, Nwx = x_shape

    ph = max(0, (Nhd - Nhx) // 2)
    pw = max(0, (Nwd - Nwx) // 2)

    return delta[:, :, ph:ph + Nhx, pw:pw + Nwx]


def create_sub_matrices(x: np.ndarray, kernel_size: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    """
    Create sub-matrices from an input array with a given kernel size and stride.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    kernel_size: tuple[int, int]
        The size of the kernel to take a max over
    stride: tuple[int, int]
        The stride of the window.

    Returns
    -------
    sub_matrices: np.ndarray
        Array of shape (batch_size, num_channels, Oh, Ow, kernel_size[0], kernel_size[1])
    """
    m, Nc, Nh, Nw = x.shape
    Oh = (Nh - kernel_size[0]) // stride[0] + 1
    Ow = (Nw - kernel_size[1]) // stride[1] + 1
    shape = (m, Nc, Oh, Ow, kernel_size[0], kernel_size[1])

    # Generate and return the sub-matrices
    item_size = x.itemsize
    strides = [Nc * Nh * Nw * item_size, Nh * Nw * item_size, Nw * stride[0] * item_size,
               stride[1] * item_size, Nw * item_size, item_size]

    # Device-agnostic code to allocate the sub-matrices
    xp = cp.get_array_module(x)
    return xp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


class IntegerAvgPool2d(Module):
    """
    Average pooling layer for images with uses integer-only arithmetic.
    Accepts 4D inputs of shape (batch_size, num_channels, height, width).

    Attributes
    ----------
    kernel_size: int | tuple[int, int]
        The size of the kernel to take a max over
    stride: int | tuple[int, int]
        The stride of the window.
        The default value is the kernel size
    padding: int | tuple[int, int]
        The amount of implicit negative infinity padding to be added to both sides of the input image
    """

    def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 0, name: str = None, dtype: str = 'int32') -> None:
        super().__init__(name=name)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        if self.kernel_size[0] != self.stride[0] or self.kernel_size[1] != self.stride[1]:
            raise NotImplementedError('Stride different than kernel size is not implemented yet')

        if self.padding[0] > self.kernel_size[0] // 2 or self.padding[1] > self.kernel_size[1] // 2:
            raise ValueError(f'Padding should be at most half of kernel size, '
                             f'but got pad={self.padding} and kernel_size={self.kernel_size}')

        self.dtype = dtype
        self.last_input_shape = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Apply padding if needed
        x_pad = add_zero_padding(x, self.padding)
        # Create sub-matrices
        sub_matrices = create_sub_matrices(x_pad, self.kernel_size, self.stride)
        # Perform the pooling operation
        return int_mean(sub_matrices, axis=(-2, -1))

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        xp = cp.get_array_module(delta)
        delta_expanded = xp.kron(delta, xp.ones((self.kernel_size[0], self.kernel_size[1]), dtype=delta.dtype))
        jh, jw = self.kernel_size[0] - self.stride[0], self.kernel_size[1] - self.stride[1]

        if jw != 0:
            L = delta_expanded.shape[-1] - 1
            l1 = xp.arange(self.stride[1], L)
            l2 = xp.arange(self.stride[1] + jw, L + jw)

            mask = xp.tile([True] * jw + [False] * jw, len(l1) // jw).astype(bool)
            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            delta_expanded[:, :, :, r1] += delta_expanded[:, :, :, r2]
            delta_expanded = xp.delete(delta_expanded, r2, axis=-1)

        if jh != 0:
            L = delta_expanded.shape[-2] - 1
            l1 = xp.arange(self.stride[0], L)
            l2 = xp.arange(self.stride[0] + jh, L + jh)

            mask = xp.tile([True] * jh + [False] * jh, len(l1) // jh).astype(bool)
            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            delta_expanded[:, :, r1, :] += delta_expanded[:, :, r2, :]
            delta_expanded = xp.delete(delta_expanded, r2, axis=-2)

        return truncated_division(delta_expanded, self.kernel_size[0] * self.kernel_size[1])

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ''
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'
        if logging_level >= 2:
            layer_repr += f', name={self.name}' if self.name else ''
            layer_repr += f', trainable={self.trainable}, debug={self.debug}, device={self.device}, dtype={self.dtype}'
        return layer_repr + ')'


class MaxPool2d(Module):
    """
    Max pooling layer for images.
    Accepts 4D inputs of shape (batch_size, num_channels, height, width).
    The backward pass is only implemented for inputs whose height and width are divisible by the kernel size for now.

    Attributes
    ----------
    kernel_size: int | tuple[int, int]
        The size of the kernel to take a max over
    stride: int | tuple[int, int]
        The stride of the window.
        The default value is the kernel size
    padding: int | tuple[int, int]
        The amount of implicit negative infinity padding to be added to both sides of the input image
    """

    def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int],
                 padding: int | tuple[int, int] = 0, name: str = None, dtype: str = 'int32') -> None:
        super().__init__(name=name)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        if self.kernel_size[0] != self.stride[0] or self.kernel_size[1] != self.stride[1]:
            raise NotImplementedError('Stride different than kernel size is not implemented yet')

        if self.padding[0] > self.kernel_size[0] // 2 or self.padding[1] > self.kernel_size[1] // 2:
            raise ValueError(f'Padding should be at most half of kernel size, '
                             f'but got pad={self.padding} and kernel_size={self.kernel_size}')

        self.dtype = dtype
        self.input_shape = None
        self.input_shape_pad = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Apply padding if needed
        x_pad = add_inf_padding(x, self.padding, self.dtype)
        self.input_shape = x.shape
        self.input_shape_pad = x_pad.shape

        # Create sub-matrices
        sub_matrices = create_sub_matrices(x_pad, self.kernel_size, self.stride)

        if self.training:
            # Create a mask to remember which element was the maximum, used in the backward pass
            a = sub_matrices.reshape(-1, self.kernel_size[0] * self.kernel_size[1])
            idx = np.argmax(a, axis=1)
            xp = cp.get_array_module(sub_matrices)
            b = xp.zeros(a.shape, dtype=self.dtype)
            b[xp.arange(len(b)), idx] = 1
            self.mask = b.reshape(sub_matrices.shape)

        # Perform the pooling operation
        return np.max(sub_matrices, axis=(-2, -1))

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        # Create an output gradient array with the same shape as the input
        xp = cp.get_array_module(delta)
        new_delta = xp.zeros(self.input_shape_pad, dtype=self.dtype)

        # Generate all input indices for each delta
        i0 = xp.repeat(xp.arange(delta.shape[2]), delta.shape[3]).reshape(delta.shape[2], delta.shape[3])
        j0 = xp.tile(xp.arange(delta.shape[3]), delta.shape[2]).reshape(delta.shape[2], delta.shape[3])

        # Calculate input indices for each kernel element
        i_indices = self.stride[0] * i0[:, :, None, None] + xp.arange(self.kernel_size[0])[None, None, :, None]
        j_indices = self.stride[1] * j0[:, :, None, None] + xp.arange(self.kernel_size[1])[None, None, None, :]

        # Expand delta for broadcasting
        expanded_delta = delta[:, :, :, :, None, None]

        # Mask selection expanded for each delta
        selected_mask = self.mask[:, :, i0, j0]
        slices = (slice(None), slice(None), i_indices, j_indices)
        xp.add.at(new_delta, slices, expanded_delta * selected_mask)
        return remove_padding(new_delta, self.input_shape)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ''
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'
        if logging_level >= 2:
            layer_repr += f', name={self.name}' if self.name else ''
            layer_repr += f', trainable={self.trainable}, debug={self.debug}, device={self.device}, dtype={self.dtype}'
        return layer_repr + ')'


class GlobalMaxPool2d(Module):
    """
    Global max pooling operation for image data.
    Accepts 4D inputs of shape (batch_size, num_channels, height, width).
    Padding and the backward pass are not yet implemented.

    Attributes
    ----------
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name=name)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=(2, 3))

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        raise NotImplementedError('Backward pass for GlobalMaxPool2d is not implemented yet')

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ''
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 2:
            layer_repr += f', name={self.name}' if self.name else ''
            layer_repr += f', trainable={self.trainable}, debug={self.debug}, device={self.device}'
        return layer_repr + ')'
