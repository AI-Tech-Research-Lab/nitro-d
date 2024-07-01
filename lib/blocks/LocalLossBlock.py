from abc import ABC
from typing import Literal

import numpy as np

from lib.layers.modules import Module
from lib.models.models import Model
from lib.utils.nn import l2_loss, accuracy, l2_loss_grad


class LocalLossBlock(Model, ABC):
    """
    Base class for all blocks that use local loss.
    The block is trained by locally generated error signal based on MSE loss.
    In debug mode, the activations and gradients of the forward and backward passes, respectively, will be saved.

    Attributes
    ----------
    local_loss: Literal['pred', 'sim', 'predsim'], default='pred'
        The type of local loss to be used
    beta: float, default=0.99
        The parameter to compute the predsim loss
    subnet_trainable: bool, default=True
        If True, the parameters of the learning layers are updated during training
    layers: Sequential
        The forward layers of the block
    last_activation: np.ndarray
        The output of the block at the previous training step, used to compute the local loss
    pred_loss_net: Sequential
        The learning layers used to compute the local loss
    """

    def __init__(self, local_loss: Literal['pred', 'sim', 'predsim'] = 'pred', beta: float = 0.99,
                 subnet_trainable: bool = True, trainable: bool = True, debug: bool = False,
                 device: Literal['cpu', 'cuda'] = 'cpu', name: str = None, **kwargs) -> None:
        if local_loss != 'pred':
            raise NotImplementedError('Only pred loss is implemented for now')

        super().__init__(debug=debug, device=device, trainable=trainable, name=name)
        self.local_loss = local_loss
        self.beta = beta
        self.subnet_trainable = subnet_trainable

        self.layers = None
        self.last_activation = None
        self.pred_loss_net = None

        self.loss_sim: float = 0.0
        self.loss: float = 0.0
        self.train_accuracy: float = 0.0

    def compute_local_loss(self, h: np.ndarray, y_onehot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the local loss and the gradient of the loss w.r.t. the output of the block.
        It also updates the internal statistics of the block.

        Parameters
        ----------
        h: np.ndarray
            The output of the block at the previous training step
        y_onehot: np.ndarray
            Ground truth labels of the input batch

        Returns
        -------
        loss_grad: tuple[np.ndarray, np.ndarray]
            A tuple of two elements containing the l2 loss and the gradient of the loss w.r.t.
            the output of the block
        """
        match self.local_loss:
            case 'sim':
                raise NotImplementedError('sim subnetwork is not implemented yet')

            case 'pred':
                # Forward pass of the learning layers
                y_hat_local = self.pred_loss_net(h)
                loss_pred = l2_loss(y_onehot, y_hat_local)
                grad_pred = l2_loss_grad(y_onehot, y_hat_local)

                # Update the internal statistics
                self.loss = loss_pred
                self.train_accuracy = accuracy(y_onehot, y_hat_local)
                return loss_pred, grad_pred

            case 'predsim':
                raise NotImplementedError('predsim subnetwork is not implemented yet')

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.layers(x)
        self.last_activation = x.copy()
        return x

    def backward(self, y_true: np.ndarray, lr_inv: int) -> np.ndarray:
        """
        Performs the backward pass of the block, computing the local loss and back-propagating it through the block.

        Parameters
        ----------
        y_true: np.ndarray
            Ground truth labels of the input batch
        lr_inv: int
            The inverse of the learning rate used to compute the parameters' updates.

        Returns
        -------
        local_loss: np.array
            The local loss of the block, i.e., the gradient of the loss w.r.t.
            the input of the block
        """
        # Forward pass of the learning layers and compute the local loss
        local_loss, delta = self.compute_local_loss(self.last_activation, y_true)

        # Back-propagate through the learning layers
        delta = self.pred_loss_net.backward(delta, lr_inv)

        # Back-propagate through the forward layers
        self.layers.backward(delta, lr_inv)
        return local_loss

    def train(self) -> None:
        self.training = True
        self.layers.train()
        if self.pred_loss_net is not None:
            self.pred_loss_net.train()

    def eval(self) -> None:
        self.training = False
        self.layers.eval()
        if self.pred_loss_net is not None:
            self.pred_loss_net.eval()

    def freeze(self) -> None:
        self.trainable = False
        self.subnet_trainable = False
        self.layers.freeze()
        if self.pred_loss_net is not None:
            self.pred_loss_net.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        self.subnet_trainable = True
        self.layers.unfreeze()
        if self.pred_loss_net is not None:
            self.pred_loss_net.unfreeze()

    def get_layers_with_parameters(self) -> list[Module]:
        layers = self.layers.get_layers_with_parameters()
        if self.pred_loss_net is not None:
            layers = layers + self.pred_loss_net.get_layers_with_parameters()
        return layers

    def get_saved_activations(self) -> dict[str, np.ndarray]:
        return self.layers.get_saved_activations()

    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        return self.layers.get_saved_gradients()

    def get_saved_subnet_activations(self) -> dict[str, np.ndarray]:
        """
        Get the saved activations of the learning layers.

        Returns
        -------
        activations: dict[str, np.ndarray]
            Dictionary with the name of the layer and the corresponding activation
        """
        activations = {}
        if self.pred_loss_net is not None:
            activations = activations | self.pred_loss_net.get_saved_activations()
        return activations

    def get_saved_subnet_gradients(self) -> dict[str, np.ndarray]:
        """
        Get the saved gradients of the learning layers.

        Returns
        -------
        gradients: dict[str, np.ndarray]
            Dictionary with the name of the layer and the corresponding gradient
        """
        gradients = {}
        if self.pred_loss_net is not None:
            gradients = gradients | self.pred_loss_net.get_saved_gradients()
        return gradients

    def freeze_subnet(self) -> None:
        """
        Freeze the parameters of the learning layers, so that they are not updated during training.

        Returns
        -------
        None
        """
        self.subnet_trainable = False
        if self.pred_loss_net is not None:
            self.pred_loss_net.freeze()

    def unfreeze_subnet(self) -> None:
        """
        Unfreeze the parameters of the learning layers, so that they are updated during training.

        Returns
        -------
        None
        """
        self.subnet_trainable = True
        if self.pred_loss_net is not None:
            self.pred_loss_net.unfreeze()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        layers = ''
        for i, layer in enumerate(self.layers.layers):
            layers += f'\t\t\t({i}): {layer.extra_repr(logging_level)}\n'
        if self.pred_loss_net is not None:
            layers += f'\t\t\t(learning_layers): {self.pred_loss_net.extra_repr(logging_level)}\n'

        return f'{self.__class__.__name__}(\n{layers}\t)'
