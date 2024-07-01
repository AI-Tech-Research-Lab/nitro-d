from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class Module(ABC):
    """
    Base class for all neural network modules.

    Attributes
    ----------
    training : bool
        Flag to indicate if the module is currently training mode or not
    trainable : bool, default=True
        Flag to indicate if the parameters of the module (if any) will be updated during training or not
    debug: bool, default=False
        Flag to indicate if the module is in debug mode or not. In debug mode, the module will save the activations
        and gradients of the forward and backward passes, respectively.
    name: str, default=None
        Name of the module, used for debugging purposes
    """

    def __init__(self, trainable: bool = True, debug: bool = False,
                 name: str = None, device: Literal['cpu', 'cuda'] = 'cpu') -> None:
        self.weights = None
        self.bias = None
        self.updates = None
        self.trainable = trainable
        self.debug = debug
        self.name = name
        self.device = device
        self.training: bool = False
        self.lr_amp_factor: int = 1

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.

        Parameters
        ----------
        x: np.array
            Input to the layer

        Returns
        -------
        result: np.array
            Output of the layer
        """
        pass

    @abstractmethod
    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        """
        Computes the backward pass of the layer.

        Parameters
        ----------
        delta: np.array
            Gradient of the loss w.r.t. the output of the layer (the gradient coming from the next layer)
        lr_inv: int
            The learning rate used to compute the parameters' updates. If the layer works with integer arithmetics,
            the learning rate is the inverse of the learning rate, i.e., lr_inv = 1 / lr

        Returns
        -------
        result: np.array
            Gradient of the loss w.r.t. the input of the layer (the gradient to be propagated to the previous layer)
        """
        pass

    def train(self) -> None:
        """
        Sets the module in training mode.

        Returns
        -------
        None
        """
        self.training = True

    def eval(self) -> None:
        """
        Sets the module in evaluation mode.

        Returns
        -------
        None
        """
        self.training = False

    def freeze(self) -> None:
        """
        Freeze the module so that its parameters (if any) will no longer be updated.

        Returns
        -------
        None
        """
        self.trainable = False

    def unfreeze(self) -> None:
        """
        Unfreeze the module so that its parameters (if any) will be updated.

        Returns
        -------
        None
        """
        self.trainable = True

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        """
        Defines the extra representation of the module, used for debugging purposes.
        It is used to print the module in a human-readable format.

        Parameters
        ----------
        logging_level: int, default=0
            Level of logging to use. If 0, only the class name of the module is printed. If 1, the class name and the
            most relevant attributes are printed. If 2, all attributes are printed

        Returns
        -------
        repr: str
            A human-readable representation of the module
        """
        return self.__class__.__name__ + '('

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Shortcut to call the forward pass of the layer.

        Parameters
        ----------
        args: list
            List of positional arguments
        kwargs: dict
            Dictionary of keyword arguments

        Returns
        -------
        result: np.array
            Output of the layer
        """
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.extra_repr(logging_level=1)
