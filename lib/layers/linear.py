from abc import ABC
from typing import Literal

import cupy as cp
import numpy as np

from lib.layers.modules import Module
from lib.utils.enums import Initialization, OptimizerName
from lib.utils.init import init_params
from lib.utils.misc import to_cpu
from lib.utils.nn import create_optimizer


class Linear(Module, ABC):
    """
    Base class for all linear layers, which apply a linear transformation of the input.

    Attributes
    ----------
    in_features: int
        Size of each input sample
    out_features: int
        Size of each output sample
    updates: list
        List of the updates of the weights of the layer during training, used for debugging
    weights: np.array
        Learnable weights of the layer of shape (out_features, in_features)
    bias: np.array
        Learnable bias of the layer of shape (1, out_features)
    """

    def __init__(self, in_features: int, out_features: int, trainable: bool = True, propagate_backward: bool = True,
                 debug: bool = False, name: str = None, device: Literal['cpu', 'cuda'] = 'cpu') -> None:
        super().__init__(trainable=trainable, name=name, debug=debug, device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.propagate_backward = propagate_backward

        self.optimizer = None
        self.weights = None
        self.bias = None

        self.updates = []
        self.last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            # Save the input for backprop
            self.last_input = x.copy()

        # Compute the linear transformation
        if self.bias is not None:
            return x @ self.weights.T + self.bias
        else:
            return x @ self.weights.T

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
        The gradient to be propagated to the previous layer is given by a matrix multiplication.

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

        new_delta = None
        if self.propagate_backward or self.debug:
            # Compute the new delta, i.e., the gradient towards the input
            new_delta = delta @ self.weights

        weight_gradient, bias_gradient = None, None
        if self.trainable or self.debug:
            # Compute the updates to the trainable parameters
            weight_gradient = delta.T @ self.last_input
            if self.bias is not None:
                bias_gradient = np.sum(delta, axis=0, keepdims=True)

        return new_delta, weight_gradient, bias_gradient

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layer_repr = ''
        if logging_level >= 0:
            layer_repr += super().extra_repr()
        if logging_level >= 1:
            layer_repr += (f'in_features={self.in_features}, out_features={self.out_features}, '
                           f'bias={self.bias is not None}')
        if logging_level >= 2:
            layer_repr += f', name={self.name}' if self.name else ''
            layer_repr += f', trainable={self.trainable}, debug={self.debug}, device={self.device}'
        return layer_repr


class IntegerLinear(Linear):
    """
    Linear Layer which uses integer-only arithmetic.

    Attributes
    ----------
    in_features: int
        Size of each input sample
    out_features: int
        Size of each output sample
    bias: bool, default=True
        Whether to learn an additive bias term of not
    decay_rate_inv: int, default=0
        The amount of weight decay to apply during parameter updates.
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    lr_amp_factor: int, default=1
        Amplification factor of the learning rate for this layer.
    dtype: str, default='int32'
        The dtype of the weights and bias.
        Can be one of ['int8', 'int16', 'int32', 'int64']
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, decay_rate_inv: int = 0,
                 name: str = None, dtype: str = 'int32', propagate_backward: bool = True, debug: bool = False,
                 init: Initialization = Initialization.UNIFORM_KAIMING_LEAKY_RELU,
                 optimizer: OptimizerName = OptimizerName.INTEGER_SGD, lr_amp_factor: int = 1,
                 device: Literal['cpu', 'cuda'] = 'cpu', trainable: bool = True) -> None:
        if dtype not in ['int8', 'int16', 'int32', 'int64']:
            raise ValueError(f'Unsupported dtype: {dtype}')

        super().__init__(in_features, out_features, trainable, propagate_backward,
                         debug=debug, name=name, device=device)
        self.dtype = dtype
        self.decay_rate_inv = decay_rate_inv
        self.lr_amp_factor = lr_amp_factor

        weight_shape = (out_features, in_features)
        bias_shape = (1, out_features) if bias else None
        xp = cp if self.device == 'cuda' else np

        self.weights, self.bias = init_params(xp, init, weight_shape, bias_shape, in_features, dtype)
        self.optimizer = create_optimizer(optimizer, self, weight_decay_inv=decay_rate_inv)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            # Save the input for backprop
            self.last_input = x.copy()

        # Compute the linear transformation
        if self.bias is not None:
            return x @ self.weights.T.astype(np.int32) + self.bias
        else:
            return x @ self.weights.T.astype(np.int32)

    def compute_gradients(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().compute_gradients(delta.astype(np.int32))

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

