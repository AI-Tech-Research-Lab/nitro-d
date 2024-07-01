from abc import ABC
from typing import Literal

import cupy as cp
import numpy as np

from lib.layers.modules import Module
from lib.utils.enums import Initialization, OptimizerName
from lib.utils.init import init_params
from lib.utils.misc import to_cpu
from lib.utils.nn import create_optimizer


def add_padding(x: np.ndarray, pad_sizes: tuple[int, int]) -> np.ndarray:
    """
    Apply zero-padding to the last two dimensions of a 4D array.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    pad_sizes: tuple[int, int]
        Number of rows and columns to pad the input with

    Returns
    -------
    x_pad: np.ndarray
        Padded array of shape (batch_size, num_channels, height + 2 * pad_size[0], width + 2 * pad_size[1])
    """
    pad_width = np.array([(0, 0), (0, 0), (pad_sizes[0], pad_sizes[0]), (pad_sizes[1], pad_sizes[1])])
    return np.pad(x, pad_width=pad_width)


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


def dilate(x: np.ndarray, dilation_rates: tuple[int, int]) -> np.ndarray:
    """
    Dilate a 4D array to perform a dilated convolution.
    Used in the backward pass when the stride of the convolution is different from 1.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    dilation_rates: tuple[int, int]
        Number of rows and columns to dilate the input with

    Returns
    -------
    x_dilated: np.ndarray
        Dilated array of shape (batch_size, num_channels, height + (height - 1) * (dilation_rate - 1),
        width + (width - 1) * (dilation_rate - 1))
    """
    Xd = np.insert(arr=x, obj=np.repeat(np.arange(1, x.shape[3]), dilation_rates[1] - 1), values=0, axis=-1)
    Xd = np.insert(arr=Xd, obj=np.repeat(np.arange(1, x.shape[2]), dilation_rates[0] - 1), values=0, axis=-2)
    return Xd


def build_sub_matrices(x: np.ndarray, kernel_shape: tuple[int, ...], stride: tuple[int, int]) -> np.ndarray:
    """
    Helper function to build sub-matrices to efficiently compute a vectorized convolution.
    Sub-matrices are allocated on the current device.

    Parameters
    ----------
    x: np.ndarray
        Input array of shape (batch_size, num_channels, height, width)
    kernel_shape: tuple[int, ...]
        Shape of the convolutional kernel
    stride: tuple[int, int]
        Stride of the convolution

    Returns
    -------
    sub_matrices: np.ndarray
        Array of shape (batch_size, num_channels, output_height, output_width, kernel_height, kernel_width)
        containing the sub-matrices to be used for the convolution
    """
    # Compute the shapes of the sub-matrices
    m, Nc, Nh, Nw = x.shape
    F, Kc, Kh, Kw = kernel_shape
    Oh = (Nh - Kh) // stride[0] + 1
    Ow = (Nw - Kw) // stride[1] + 1

    # Generate and return the sub-matrices
    item_size = x.itemsize
    strides = [Nc * Nh * Nw * item_size, Nw * Nh * item_size, Nw * stride[0] * item_size,
               stride[1] * item_size, Nw * item_size, item_size]

    # Device-agnostic code to allocate the sub-matrices
    xp = cp.get_array_module(x)
    sub_matrices = xp.lib.stride_tricks.as_strided(x, shape=(m, Nc, Oh, Ow, Kh, Kw), strides=strides)
    return sub_matrices


class Conv2d(Module, ABC):
    """
    Base class for all convolutional 2d layers.

    Attributes
    ----------
    in_channels: int
        Number of channels of the input image
    out_channels: int
        Number of channels produced by the convolution.
        It corresponds to the number of filters applied to the input image
    kernel_size: int
        Size of the convolutional kernel
    stride: int, default=1
        Stride of the convolution
    padding: int, default=1
       Padding added to all four sides of the input.
       The default is 1, which with 3x3 filters gives an output of the same size as the input
    weights: np.ndarray
        Learnable weights of the layer of shape (out_channels, in_channels, kernel_size, kernel_size)
    bias: np.ndarray
        Learnable bias of the layer of shape (1, out_channels, 1, 1)
    propagate_backward: bool, default=True
        Whether to propagate the error backwards to the previous layer.
        If set to False, the computation of the gradient of the input is skipped.
        This is because it is not always required with local loss.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 1, propagate_backward: bool = True, name: str = None, debug: bool = False,
                 device: Literal['cpu', 'cuda'] = 'cpu', trainable: bool = True) -> None:
        if stride != 1:
            raise NotImplementedError('Stride different from 1 is not supported yet')

        super().__init__(trainable=trainable, name=name, debug=debug, device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.propagate_backward = propagate_backward

        self.optimizer = None
        self.weights = None
        self.bias = None

        self.updates = []
        self.last_input_padded = None
        self.last_input_shape = None

        self.forward_path = None
        self.backward_path = None
        self.params_path = None

    def conv2d_forward(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        """
        Functional to compute a forward Conv2d operation between a 4D input and a 4D kernel.

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (batch_size, num_channels, height, width)
        kernel: np.ndarray
            Kernel array of shape (num_filters, num_channels, kernel_height, kernel_width)
        stride: tuple[int, int]
            Stride of the convolution

        Returns
        -------
        result: np.ndarray
            Output of the convolution of shape (batch_size, num_filters, output_height, output_width)
        """
        sub_matrices = build_sub_matrices(x, kernel.shape, stride)
        if self.forward_path is None:
            # Pre-compute and save the optimal path for the einsum operation
            self.forward_path = np.einsum_path(
                'fckl,mcijkl->mfij',
                to_cpu(kernel), to_cpu(sub_matrices),
                optimize='optimal')[0]
        return np.einsum('fckl,mcijkl->mfij', kernel, sub_matrices, optimize=self.forward_path)

    def conv2d_backward(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        """
        Functional to compute a backward Conv2d operation between a 4D input and a 4D kernel.

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (batch_size, num_channels, height, width)
        kernel: np.ndarray
            Kernel array of shape (num_filters, num_channels, kernel_height, kernel_width)
        stride: tuple[int, int]
            Stride of the convolution

        Returns
        -------
        result: np.ndarray
            Output of the convolution of shape (batch_size, num_filters, output_height, output_width)
        """
        sub_matrices = build_sub_matrices(x, kernel.shape, stride)
        if self.backward_path is None:
            # Pre-compute and save the optimal path for the einsum operation
            self.backward_path = np.einsum_path(
                'fdkl,mcijkl->mdij',
                to_cpu(kernel), to_cpu(sub_matrices),
                optimize='optimal')[0]
        return np.einsum('fdkl,mcijkl->mdij', kernel, sub_matrices, optimize=self.backward_path)

    def conv2d_params(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        """
        Functional to compute a Conv2d operation meant to update parameters between a 4D input and a 4D kernel.

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (batch_size, num_channels, height, width)
        kernel: np.ndarray
            Kernel array of shape (num_filters, num_channels, kernel_height, kernel_width)
        stride: tuple[int, int]
            Stride of the convolution

        Returns
        -------
        result: np.ndarray
            Output of the convolution of shape (batch_size, num_filters, output_height, output_width)
        """
        sub_matrices = build_sub_matrices(x, kernel.shape, stride)
        if self.params_path is None:
            # Pre-compute and save the optimal path for the einsum operation
            self.params_path = np.einsum_path(
                'mfkl,mcijkl->fcij',
                to_cpu(kernel), to_cpu(sub_matrices),
                optimize='optimal')[0]
        return np.einsum('mfkl,mcijkl->fcij', kernel, sub_matrices, optimize=self.params_path)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Apply padding if needed
        x_pad = add_padding(x, self.padding)

        if self.training:
            # Need to save the original input's shape and the padded input values for backpropagation
            self.last_input_shape = x.shape
            self.last_input_padded = x_pad

        if self.bias is not None:
            return self.conv2d_forward(x_pad, self.weights, self.stride) + self.bias
        else:
            return self.conv2d_forward(x_pad, self.weights, self.stride)

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        new_delta, weight_gradient, bias_gradient = self.compute_gradients(delta)

        if self.trainable:
            weight_update, bias_update = self.optimizer.compute_updates(lr_inv, weight_gradient, bias_gradient)

            # Perform the update
            self.weights = self.weights - weight_update
            if self.bias is not None:
                self.bias = self.bias - bias_update

            # Save the values of the updates for debugging
            if self.debug:
                self.updates.append(to_cpu(weight_update))

        # Back-propagate the error to the next layer
        return new_delta

    def train(self) -> None:
        super().train()
        self.updates.clear()

    def compute_gradients(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The gradient to be propagated to the previous layer is a convolution with stride=1 of:
         - The padded and dilated (according to the stride) gradient from the next layer (delta)
         - A 180 degrees rotated version of the kernel

        Parameters
        ----------
        delta: np.ndarray
            Gradient of the loss w.r.t. the output of the layer (the gradient coming from the next layer)

        Returns
        -------
        gradients: tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple of three elements containing the gradient of the loss w.r.t. the input of the layer
           (i.e., the gradient to be propagated to the previous layer), the gradient of the loss w.r.t.
           the weights (i.e., the weight update), and the gradient of the loss w.r.t. the bias (i.e., the bias update)

        """
        # Dilate the delta according to the stride
        # delta_dilated = dilate(delta, self.stride)
        delta_dilated = delta

        # Pad delta_dilated to make it the same size as x_pad
        m, F, Hd, Wd = delta_dilated.shape
        m, Nc, Nh, Nw = self.last_input_padded.shape
        F, Kc, Kh, Kw = self.weights.shape

        new_delta = None
        if self.propagate_backward or self.debug:
            if not self.debug:
                raise NotImplementedError('Input gradient computation for Conv2d layer is not supported yet')
            # Compute the new delta, i.e., the gradient towards the input
            ph = ((Nh - self.padding[0]) - Hd + Kh - 1) // 2
            pw = ((Nw - self.padding[1]) - Wd + Kw - 1) // 2
            delta_pad = add_padding(delta_dilated, (ph, pw))

            # Rotate the kernel by 180 degrees
            kernel_rotated = self.weights[:, :, ::-1, ::-1]

            # Compute the backward convolution
            delta_back = self.conv2d_backward(delta_pad, kernel_rotated, stride=(1, 1))
            new_delta = remove_padding(delta_back, self.last_input_shape)

        weight_gradient, bias_gradient = None, None
        if self.trainable or self.debug:
            # Compute the updates to the trainable parameters
            ph = Nh - Hd - Kh + 1
            pw = Nw - Wd - Kw + 1
            delta_pad = add_padding(delta_dilated, (ph, pw))
            weight_gradient = self.conv2d_params(self.last_input_padded, delta_pad, stride=(1, 1))
            if self.bias is not None:
                bias_gradient = np.sum(delta, axis=(0, 2, 3), keepdims=True)

        return new_delta, weight_gradient, bias_gradient

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ''
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                           f'bias={self.bias is not None}')
        if logging_level >= 2:
            layer_repr += f', name={self.name}' if self.name else ''
            layer_repr += f', trainable={self.trainable}, debug={self.debug}, device={self.device}'
        return layer_repr


class IntegerConv2d(Conv2d):
    """
    Conv2d layer which uses integer-only arithmetic.

    Attributes
    ----------
    in_channels: int
        Number of channels of the input image
    out_channels: int
        Number of channels produced by the convolution.
        It corresponds to the number of filters applied to the input image
    kernel_size: int
        Size of the convolutional kernel
    stride: int, default=1
        Stride of the convolution
    padding: int, default=1
       Padding added to all four sides of the input.
       The default is 1, which with 3x3 filters gives an output of the same size as the input
    bias: bool, default=True
        Whether to learn an additive bias term of note
    dtype: str, default='int32'
        The dtype of the weights and bias.
        Can be one of ['int8', 'int16', 'int32', 'int64']
    decay_rate_inv: int, default=0
        The amount of weight decay to apply during parameter updates.
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    lr_amp_factor: int, default=1
        Amplification factor of the learning rate for this layer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 bias: bool = True, propagate_backward: bool = True, name: str = None, dtype: str = 'int32',
                 optimizer: OptimizerName = OptimizerName.INTEGER_SGD, debug: bool = False, decay_rate_inv: int = 0,
                 device: Literal['cpu', 'cuda'] = 'cpu', trainable: bool = True, lr_amp_factor: int = 1,
                 init: Initialization = Initialization.UNIFORM_KAIMING_LEAKY_RELU) -> None:
        if dtype not in ['int8', 'int16', 'int32', 'int64']:
            raise ValueError(f'Unsupported dtype: {dtype}')

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, propagate_backward,
                         name=name, debug=debug, device=device, trainable=trainable)
        self.dtype = dtype
        self.decay_rate_inv = decay_rate_inv
        self.lr_amp_factor = lr_amp_factor

        in_features = in_channels * kernel_size * kernel_size
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        bias_shape = (1, out_channels, 1, 1) if bias else None
        xp = cp if self.device == 'cuda' else np

        self.weights, self.bias = init_params(xp, init, weight_shape, bias_shape, in_features, dtype)
        self.optimizer = create_optimizer(optimizer, self, weight_decay_inv=decay_rate_inv)

    def conv2d_forward(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        return super().conv2d_forward(x, kernel.astype(np.int32), stride)

    def conv2d_backward(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        return super().conv2d_backward(x, kernel.astype(np.int32), stride)

    def conv2d_params(self, x: np.ndarray, kernel: np.ndarray, stride: tuple[int, int]) -> np.ndarray:
        return super().conv2d_params(x, kernel.astype(np.int32), stride)

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        if logging_level == 0:
            return super().extra_repr(logging_level) + ')'
        if logging_level == 1:
            return super().extra_repr(logging_level) + f', decay_rate_inv={self.decay_rate_inv})'
        if logging_level == 2:
            return super().extra_repr(logging_level) + (f', decay_rate_inv={self.decay_rate_inv}, '
                                                        f'dtype={self.dtype}, '
                                                        f'optimizer={self.optimizer.__class__.__name__}, '
                                                        f'lr_amp_factor={self.lr_amp_factor})')
