from typing import Literal

import numpy as np

from lib.layers.modules import Module
from lib.models.models import Model


class Sequential(Model):
    """
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    The forward pass of Sequential will call the forward pass of each module in the correct order.
    The same applies for the backward pass.
    In debug mode, the activations and gradients of the forward and backward passes, respectively, will be saved.

    Attributes
    ----------
    layers: list[Module, ...]
        List of arbitrary length of modules to be executed sequentially
    activations: list[np.ndarray, ...]
        List of activations of the forward pass of the layer. It is used for debugging purposes
    gradients: list[np.ndarray, ...]
        List of gradients of the backward pass of the layer. It is used for debugging purposes
    """

    def __init__(self, layers: list[Module], trainable: bool = True, debug: bool = True,
                 name: str = None, device: Literal['cpu', 'cuda'] = 'cpu') -> None:
        super().__init__(trainable=trainable, debug=debug, name=name, device=device)
        self.layers = layers
        self.activations: list = [0 for _ in range(len(layers))]
        self.gradients: list = [0 for _ in range(len(layers))]

    def save_activation(self, index: int, x: np.ndarray) -> None:
        """
        Saves the activation of the forward pass of the layer, only during training and in debug mode.

        Parameters
        ----------
        index: int
            Index of the layer to save the activation
        x: np.ndarray
            Activation to save

        Returns
        -------
        None
        """
        if self.debug and self.training:
            self.activations[index] = x.copy()

    def save_gradient(self, index: int, delta: np.ndarray) -> None:
        """
        Saves the gradient of the backward pass of the layer, only during training and in debug mode.

        Parameters
        ----------
        index: int
            Index of the layer to save the gradient
        delta: np.ndarray
            Gradient to save

        Returns
        -------
        None
        """
        if self.debug:
            self.gradients[index] = delta.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.save_activation(i, x)
        return x

    def backward(self, delta: np.ndarray, lr_inv: int) -> np.ndarray:
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.backward(delta, lr_inv)
            self.save_gradient(i, delta)
        return delta

    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            layer.eval()

    def freeze(self) -> None:
        self.trainable = False
        for layer in self.layers:
            layer.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        for layer in self.layers:
            layer.unfreeze()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layers = ''
        for i, layer in enumerate(self.layers):
            layers += f'\t\t\t({i}): {layer.extra_repr(logging_level)}\n'
        return f'{super().extra_repr(logging_level)}\n{layers}\t\t)'

    def get_layers_with_parameters(self) -> list[Module]:
        return [layer for layer in self.layers if layer.weights is not None]

    def get_saved_activations(self) -> dict[str, np.ndarray]:
        return {self.layers[i].name: self.activations[i] for i in range(len(self.layers))}

    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        return {self.layers[i].name: self.gradients[i] for i in range(len(self.layers))}

    def __getitem__(self, item: int) -> Module:
        return self.layers[item]
