from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from tqdm.auto import trange

from lib.layers.conv import Conv2d
from lib.layers.linear import Linear
from lib.layers.modules import Module
from lib.utils.enums import NonLinearity


class Model(Module, ABC):
    """
    Base class for all models.
    Forward and backward passes can be freely overridden.
    In debug mode, the activations and gradients of the forward and backward passes, respectively, will be saved.

    Attributes
    ----------
    """

    def __init__(self, debug: bool = False, name: str = None, trainable: bool = True,
                 device: Literal['cpu', 'cuda'] = 'cpu') -> None:
        super().__init__(trainable=trainable, name=name, debug=debug, device=device)

    @abstractmethod
    def get_layers_with_parameters(self) -> list[Module]:
        """
        Returns a list with references to the layers that have parameters (i.e., weights and biases) in the module.
        It is used for debugging purposes.

        Returns
        -------
        trainable_layers: list[Module]
            List of references to the layers with parameters
        """
        pass

    def print_layers_parameters(self) -> None:
        """
        Prints the parameters of the layers of the model.

        Returns
        None
        -------
        """
        param_layers = self.get_layers_with_parameters()
        for layer in param_layers:
            print(f'Layer: {layer.name}')
            if layer.weights is not None:
                print(f'- Weights:\n{layer.weights}')
            if layer.bias is not None:
                print(f'- Bias:\n{layer.bias}\n')

    def get_linear_layers(self) -> list[Module]:
        """
        Returns a list with references to the linear layers of the model.

        Returns
        -------
        trainable_layers: list[Module]
            List of references to the linear layers
        """
        linear_layers = []
        for layer in self.get_layers_with_parameters():
            if isinstance(layer, Linear):
                linear_layers.append(layer)
        return linear_layers

    def get_conv_layers(self) -> list[Module]:
        """
        Returns a list with references to the convolutional layers of the model.

        Returns
        -------
        trainable_layers: list[Module]
            List of references to the convolutional layers
        """
        conv_layers = []
        for layer in self.get_layers_with_parameters():
            if isinstance(layer, Conv2d):
                conv_layers.append(layer)
        return conv_layers

    def predict(self, x: np.ndarray, batch_size: int = 128, progress_bar: bool = True) -> np.ndarray:
        """
        Generates output predictions for the input samples.
        Computation is done in batches, which are automatically generated.

        Parameters
        ----------
        x : np.ndarray
            Input data
        batch_size : int, default=128
            Batch size to be used for the prediction
        progress_bar : bool, default=True
            Whether to show a progress bar or not

        Returns
        -------
        np.ndarray
            Predictions of the model for the provided input data
        """
        predictions = []
        self.eval()
        for i in trange(0, len(x), batch_size, disable=(not progress_bar)):
            x_batch = x[i:i + batch_size]
            predictions.append(self.forward(x_batch))
        return np.concatenate(predictions)

    @abstractmethod
    def get_saved_activations(self) -> dict[str, np.ndarray]:
        """
        Return the activations of the model's forward pass, saved during the last training step.

        Returns
        -------
        activations: dict[str, np.ndarray]
            Dictionary with the name of the layer and the corresponding activation
        """
        pass

    @abstractmethod
    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        """
        Return the gradients of the model's backward pass, saved during the last training step.

        Returns
        -------
        activations: dict[str, np.ndarray]
            Dictionary with the name of the layer and the corresponding gradient
        """
        pass


class LocalLossModel(Model, ABC):
    """
    Base class for models that can be trained by locally generated error signals.

    Attributes
    ----------
    blocks: List[LocalLossBlock]
        List of the local loss blocks of the model, trained with local BP
    layers: Sequential
        Forward layers of the model, wrapped in a Sequential module
    num_classes : int
        Number of classes of the classification task
    non_linearity : Enum
        Non-linear activation function to be used in the model
    subnet_trainable: bool, default=True
        Whether the learning layers of the local loss blocks are trainable or not
    """

    def __init__(self, num_classes: int, non_linearity: NonLinearity, subnet_trainable: bool = True,
                 local_loss: Literal['pred', 'sim', 'predsim'] = 'pred', beta: float = 0.99, debug: bool = False,
                 device: Literal['cpu', 'cuda'] = 'cpu', trainable: bool = True, name: str = None) -> None:
        super().__init__(trainable=trainable, name=name, debug=debug, device=device)
        self.num_classes = num_classes
        self.non_linearity = non_linearity
        self.subnet_trainable = subnet_trainable
        self.local_loss = local_loss
        self.beta = beta

        self.blocks = []
        self.layers = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.layers(x)

    @abstractmethod
    def backward(self, delta: np.ndarray, lr_inv: int, y_true: np.ndarray = None) -> np.ndarray:
        pass

    def train(self) -> None:
        self.training = True
        self.layers.train()

    def eval(self) -> None:
        self.training = False
        self.layers.eval()

    def freeze(self) -> None:
        self.trainable = False
        self.subnet_trainable = False
        self.layers.freeze()

    def unfreeze(self) -> None:
        self.trainable = True
        self.subnet_trainable = True
        self.layers.unfreeze()

    def get_layers_with_parameters(self) -> list:
        return self.layers.get_layers_with_parameters()

    def get_saved_activations(self) -> dict[str, np.ndarray]:
        return self.layers.get_saved_activations()

    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        return self.layers.get_saved_gradients()

    def get_saved_subnets_activations(self) -> dict[str, np.ndarray]:
        activations = {}
        for block in self.blocks:
            activations = activations | block.get_saved_subnet_activations()
        return activations

    def get_saved_subnets_gradients(self) -> dict[str, np.ndarray]:
        gradients = {}
        for block in self.blocks:
            gradients = gradients | block.get_saved_subnet_gradients()
        return gradients

    def freeze_subnets(self) -> None:
        self.subnet_trainable = False
        for block in self.blocks:
            block.freeze_subnet()

    def unfreeze_subnets(self) -> None:
        self.subnet_trainable = True
        for block in self.blocks:
            block.unfreeze_subnet()

    def subnetworks_predict(self, x: np.ndarray, batch_size: int = 128, progress_bar: bool = False) -> list[np.ndarray]:
        """
        Generates output predictions for the input samples, using the learning layers of the local loss blocks.
        It returns a tensor of shape (num_blocks, num_samples, num_classes) containing the predictions.
        Computation is done in batches, which are automatically generated.

        Parameters
        ----------
        x : np.ndarray
            Input data
        batch_size : int, default=128
            Batch size to be used for the prediction
        progress_bar : bool, default=True
            Whether to show a progress bar or not

        Returns
        -------
        list: np.ndarray
            List of predictions of the learning layers of local loss blocks for the provided input data
        """
        subnets_predictions = [[] for _ in range(len(self.blocks))]
        for i in trange(0, len(x), batch_size, disable=(not progress_bar)):
            x_batch = x[i:i + batch_size]
            # Model.forward saves the activations of the blocks
            self.forward(x_batch)
            for j, block in enumerate(self.blocks):
                # Iterate through the blocks and get the predictions of the learning layers
                y_pred = block.pred_loss_net(block.last_activation)
                subnets_predictions[j].append(y_pred)
        return [np.concatenate(predictions) for predictions in subnets_predictions]
