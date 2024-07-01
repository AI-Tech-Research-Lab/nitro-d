from typing import Literal, Sequence

import numpy as np

from lib.blocks.LocalConv2dBlock import LocalConv2dBlock
from lib.blocks.LocalLinearBlock import LocalLinearBlock
from lib.layers.linear import IntegerLinear
from lib.layers.misc import Flatten, Dropout, ConstantScaling
from lib.layers.pooling import MaxPool2d, IntegerAvgPool2d, GlobalMaxPool2d
from lib.models.models import LocalLossModel
from lib.models.sequential import Sequential
from lib.utils.enums import NonLinearity, Initialization, OptimizerName


class IntegerLocalLossMLP(LocalLossModel):
    """
    Integer-only Multi-Layer Perceptron (MLP) that can be trained by locally generated error signals.
    It is composed of num_layers integer local-loss blocks and by a regular Integer Linear layer which is the output.

    Attributes
    ----------
    num_classes : int
        Number of classes of the classification task
    non_linearity : Enum
        Non-linear activation function to be used in the model
    num_layers : int
        Number of layers of the model: (num_layers - 1) blocks + 1 regular output layer
    num_hidden_neurons : tuple[int]
        A tuple of integers representing the number of neurons in each hidden layer
    fwd_decay_inv: int, default=0
        The amount of weight decay to apply during parameter updates in the forward part of the blocks
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    subnet_decay_inv: int, default=0
        The amount of weight decay to apply during parameter updates in the learning layers of the blocks
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    lr_amp_factor: int, default=None
        Amplification factor of the learning rate to be used in local loss blocks
        If None, it is computed automatically.
    layers: Sequential
        Forward layers of the model, wrapped in a Sequential module
    input: Flatten
        Input layer of the model which flattens the input image into a vector
    blocks: List[LocalLossBlock]
        List of the local loss blocks of the model, trained with local BP
    output_layers: Sequential
        Output layers of the model wrapped in a Sequential module, trained with regular BP
    """

    def __init__(self, num_fc_layers: int, num_fc_hidden: tuple[int], num_classes: int, input_dim: int,
                 dropout_rate: float, non_linearity: NonLinearity, local_loss: Literal['pred', 'sim', 'predsim'],
                 optimizer: OptimizerName, dtype: str = 'int32', name_index: int = 0, debug: bool = False,
                 bias: bool = True, fwd_decay_inv: int = 0, subnet_decay_inv: int = 0,
                 device: Literal['cpu', 'cuda'] = 'cpu', name: str = None, trainable: bool = True, beta: float = 0.99,
                 subnet_trainable: bool = True, lr_amp_factor: int = None,
                 init: Initialization = Initialization.UNIFORM_STD) -> None:
        if (num_fc_layers - 1) != len(num_fc_hidden):
            raise ValueError('The length of num_hidden_neurons must be num_layers - 1')
        super().__init__(num_classes, non_linearity, subnet_trainable, local_loss, beta,
                         name=name, debug=debug, device=device, trainable=trainable)
        self.num_layers = num_fc_layers
        self.num_hidden_neurons = num_fc_hidden
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.fwd_decay_inv = fwd_decay_inv
        self.subnet_decay_inv = subnet_decay_inv
        self.optimizer = optimizer

        # Compute the suggested amplification factor of the learning rate for the first layer
        self.lr_amp_factor = (2 ** 6) * num_classes if lr_amp_factor is None else lr_amp_factor

        # Instantiate layers
        self.input = Flatten(name=f'flatten_{name_index:02d}')
        self.hidden_layers = []
        self.blocks = []

        # If more two or more layers are requested, instantiate local loss blocks
        for i in range(0, num_fc_layers - 1):
            in_features = (input_dim if i == 0 else num_fc_hidden[i - 1])
            self.hidden_layers.append(
                LocalLinearBlock(
                    in_features=in_features, out_features=num_fc_hidden[i], num_classes=num_classes,
                    non_linearity=non_linearity, optimizer=optimizer,
                    name=f'linear_block_{i + name_index:02d}', fwd_decay_inv=fwd_decay_inv,
                    subnet_decay_inv=subnet_decay_inv, debug=debug, bias=bias, dtype=dtype,
                    name_index=i + name_index, init=init, lr_amp_factor=self.lr_amp_factor, beta=beta,
                    trainable=trainable, local_loss=local_loss, device=device,
                    subnet_trainable=subnet_trainable
                )
            )

            # Keep track of a reference to the block
            self.blocks.append(self.hidden_layers[-1])

            if self.dropout_rate != 0.0:
                self.hidden_layers.append(Dropout(dropout_rate=self.dropout_rate, name=f'dropout_{i + name_index:02d}'))

        # The output is always given a by a regular linear layer
        output_layers = []
        in_features = (input_dim if num_fc_layers == 1 else num_fc_hidden[-1])
        output_linear = IntegerLinear(
            in_features=in_features, out_features=num_classes, bias=bias,
            decay_rate_inv=subnet_decay_inv,
            name=f'linear_{len(self.hidden_layers) + name_index:02d}', dtype=dtype,
            propagate_backward=False, debug=debug, init=init, optimizer=optimizer,
            device=device, trainable=trainable
        )
        output_layers.append(output_linear)
        output_layers.append(
            ConstantScaling(
                fwd_factor=(2 ** 8) * in_features,
                name=f'scaling_{len(self.hidden_layers) + name_index:02d}'
            )
        )

        # Connect all the layers through a Sequential module for convenience
        self.output_layers = Sequential(output_layers, debug=debug, device=device, trainable=trainable)
        self.layers = Sequential([self.input] + self.hidden_layers + [self.output_layers],
                                 debug=debug, device=device, trainable=trainable)

    def backward(self, delta: np.ndarray, lr_inv: int, y_true: np.ndarray = None) -> np.ndarray:
        # Backward pass of the output layers, with regular BP
        new_delta = self.output_layers.backward(delta, lr_inv)
        # Backward pass of the local loss blocks, with local BP
        for block in reversed(self.blocks):
            new_delta = block.backward(y_true, lr_inv)
        # Return the gradient of the input for debugging purposes
        return new_delta

    def get_saved_activations(self) -> dict[str, np.ndarray]:
        activations = {}
        # Activation of the Flatten layer
        activations = activations | {self.input.name: self.layers.activations[0]}
        # Activations of the local loss blocks
        for block in self.blocks:
            activations = activations | block.get_saved_activations()
        # Activations of the output layers
        activations = activations | self.output_layers.get_saved_activations()
        return activations

    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        gradients = {}
        # Gradient of the Flatten layer
        gradients = gradients | {self.input.name: self.layers.gradients[0]}
        # Gradients of the local loss blocks
        for block in self.blocks:
            gradients = gradients | block.get_saved_gradients()
        # Gradients of the output layers
        gradients = gradients | self.output_layers.get_saved_gradients()
        return gradients

    def get_layers_with_parameters(self) -> list:
        param_layers = []
        for block in self.blocks:
            param_layers = param_layers + block.get_layers_with_parameters()
        return param_layers + self.output_layers.get_layers_with_parameters()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        return f'{self.__class__.__name__}(\n\t(0): {self.layers.extra_repr(logging_level)}\n\t)'


class IntegerLocalLossCNN(LocalLossModel):
    """
    Integer-only Convolutional Neural Network that can be trained by locally generated error signals.
    The feature extraction part is entirely specified by the fe_architecture parameter.
    The dense classifier is an IntegerLocalLossMLP model composed of num_fc_layers blocks.

    Attributes
    ----------
    num_classes : int
        Number of classes of the classification task
    non_linearity : Enum
        Non-linear activation function to be used in the model
    input_channels : int
        Number of channels of the input
    image_size : int
        Size of the input image
    fe_architecture: Sequence[tuple]
        Architecture of the feature extractor, represented as a tuple of tuples
        Each tuple represents a layer, and is composed of:
        - Layer type: 'Conv2d', 'MaxPool2d', 'AvgPool2d'
        - Layer parameters: num_filters, kernel_size, stride, padding
    feature_extractor: Sequential
        Feature extractor of the model, wrapped in a Sequential module
    classifier: Sequential
        Classifier of the model, wrapped in a Sequential module
    num_fc_layers: int
        Number of blocks of the dense classifier
    num_hidden_fc: tuple[int]
        A tuple of integers representing the number of neurons of each hidden layer
    lr_amp_factor: int, default=None
        Amplification factor of the learning rate to be used in local loss blocks
        If None, it is computed automatically.
    fe_dropout_rate: float, default=0.0
        Dropout rate to be used in the feature extractor
    fc_dropout_rate: float, default=0.0
        Dropout rate to be used in the dense classifier
    bias: bool, default=True
        Whether to learn an additive bias term of not
    fwd_decay_inv: int, default=0
        The amount of weight decay to apply during parameter updates in the forward part of the blocks
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    subnet_decay_inv: int, default=0
        The amount of weight decay to apply during parameter updates in the learning layers of the blocks
        When equal to 0, no weight decay is applied.
        With larger numbers, the weight decay is smaller.
    layers: Sequential
        Forward layers of the model, wrapped in a Sequential module
    blocks: List[LocalLossBlock]
        List of the local loss blocks of the model, trained with local BP
    feature_extractor: Sequential
        Feature extractor of the model, wrapped in a Sequential module
    classifier: IntegerLocalLossMLP
        Final dense classifier of the model
    subnet_pooling_type: str, default='max'
        Type of pooling to be used in the learning layers
    adjust_conv_amp: bool, default=True
        Whether to adjust the amplification factor of Conv2d layers based on the input size and pooling size
    """

    def __init__(self, input_channels: int, image_size: int, fe_architecture: Sequence[tuple], optimizer: OptimizerName,
                 num_fc_layers: int, num_hidden_fc: tuple[int], num_classes: int, non_linearity: NonLinearity,
                 local_loss: Literal['pred', 'sim', 'predsim'], subnet_pooling_type: Literal['max', 'avg'] = 'max',
                 fe_dropout_rate: float = 0.0, fc_dropout_rate: float = 0.0, pred_decoder_dim: int = 2048,
                 debug: bool = False, bias: bool = True, name: str = None, trainable: bool = True, beta: float = 0.99,
                 subnet_trainable: bool = True, dtype: str = 'int32', subnet_decay_inv: int = 0, fwd_decay_inv: int = 0,
                 device: Literal['cpu', 'cuda'] = 'cpu', lr_amp_factor: int = None, adjust_conv_amp: bool = True,
                 init: Initialization = Initialization.UNIFORM_STD) -> None:
        super().__init__(num_classes, non_linearity, subnet_trainable, local_loss, beta,
                         name=name, debug=debug, device=device, trainable=trainable)
        self.input_channels = input_channels
        self.image_size = image_size
        self.fe_architecture = fe_architecture
        self.num_fc_layers = num_fc_layers
        self.num_hidden_fc = num_hidden_fc
        self.fe_dropout_rate = fe_dropout_rate
        self.fc_dropout_rate = fc_dropout_rate
        self.pred_decoder_dim = pred_decoder_dim
        self.subnet_decay_inv = subnet_decay_inv
        self.fwd_decay_inv = fwd_decay_inv
        self.bias = bias
        self.dtype = dtype
        self.init = init
        self.optimizer = optimizer
        self.subnet_pooling_type = subnet_pooling_type
        self.adjust_conv_amp = adjust_conv_amp

        # Compute the suggested amplification factor of the learning rate for the first layer
        self.lr_amp_factor = (2 ** 6) * num_classes if lr_amp_factor is None else lr_amp_factor

        # Instantiate the feature extractor
        fe_layers, dim_out, channels_out = self.__make_fe_layers(fe_architecture)
        # Adjust the last Dropout layer if needed
        if self.fe_dropout_rate == 0 and self.fc_dropout_rate != 0.0:
            fe_layers.insert(-1, Dropout(dropout_rate=self.fc_dropout_rate, name=f'dropout_{len(fe_layers):02d}'))
        elif self.fe_dropout_rate != 0.0 and self.fc_dropout_rate != 0:
            assert isinstance(fe_layers[-2], Dropout)
            fe_layers[-2].dropout_rate = self.fc_dropout_rate
        self.feature_extractor = Sequential(fe_layers, debug=debug, device=device, trainable=trainable)

        # Instantiate the MLP final classifier
        self.classifier = IntegerLocalLossMLP(
            num_fc_layers=num_fc_layers, num_fc_hidden=num_hidden_fc,
            num_classes=num_classes, input_dim=dim_out * dim_out * channels_out,
            dropout_rate=fc_dropout_rate, non_linearity=non_linearity,
            local_loss=local_loss, optimizer=optimizer, dtype=dtype,
            name_index=len(fe_architecture), debug=debug, bias=bias,
            fwd_decay_inv=fwd_decay_inv, subnet_decay_inv=subnet_decay_inv,
            device=device, name=f'local_loss_mlp', trainable=trainable, beta=beta,
            subnet_trainable=subnet_trainable, lr_amp_factor=self.lr_amp_factor,
            init=init
        )

        # Keep a reference to local loss blocks in the MLP
        self.blocks += self.classifier.blocks

        # Connect all the layers through a Sequential module for convenience
        self.layers = Sequential([self.feature_extractor, self.classifier],
                                 debug=debug, device=device, trainable=trainable)

        if self.adjust_conv_amp:
            # Adjust the amplification factor for LocalConv2dBlocks considering input size and pooling size
            adjustments = []
            for block in self.blocks:
                if isinstance(block, LocalConv2dBlock):
                    # Compute the adjustment for each Conv2d layer
                    adjustment = (block.dim_out ** 2) // (block.ks_h * block.ks_w)
                    adjustments.append(adjustment)

            # min_adjustment = np.min(adjustments)
            for i, block in enumerate(self.blocks):
                if isinstance(block, LocalConv2dBlock):
                    # Apply the adjustment to the amplification factor
                    actual_adjustment = adjustments[i]
                    block.lr_amp_factor *= actual_adjustment
                    block.layers.layers[0].lr_amp_factor *= actual_adjustment

    def backward(self, delta: np.ndarray, lr_inv: int, y_true: np.ndarray = None) -> np.ndarray:
        # Backward pass of the dense classifier
        new_delta = self.classifier.backward(delta, lr_inv, y_true)
        # Backward pass of the feature extractor, skipping pooling layers
        for block in reversed(self.blocks):
            new_delta = block.backward(y_true, lr_inv)
        # Return the gradient of the input for debugging purposes
        return new_delta

    def get_saved_activations(self) -> dict[str, np.ndarray]:
        activations = {}
        # Activations of the feature extractor
        for i, layer in enumerate(self.feature_extractor.layers):
            if isinstance(layer, LocalConv2dBlock):
                activations = activations | layer.get_saved_activations()
            else:
                activations = activations | {layer.name: self.feature_extractor.activations[i]}
        # Activations of the dense classifier
        activations = activations | self.classifier.get_saved_activations()
        return activations

    def get_saved_gradients(self) -> dict[str, np.ndarray]:
        gradients = {}
        # Gradients of the feature extractor
        for i, layer in enumerate(self.feature_extractor.layers):
            if isinstance(layer, LocalConv2dBlock):
                gradients = gradients | layer.get_saved_gradients()
            else:
                gradients = gradients | {layer.name: self.feature_extractor.gradients[i]}
        # Gradients of the dense classifier
        gradients = gradients | self.classifier.get_saved_gradients()
        return gradients

    def get_layers_with_parameters(self) -> list:
        param_layers = []
        for block in self.blocks:
            param_layers = param_layers + block.get_layers_with_parameters()
        return param_layers + self.classifier.output_layers.get_layers_with_parameters()

    def extra_repr(self, logging_level: Literal[0, 1, 2] = 0) -> str:
        return (f'{self.__class__.__name__}(\n\t(0): {self.feature_extractor.extra_repr(logging_level)}\n\t(1): '
                f'{self.classifier.extra_repr(logging_level)}\n)')

    def __make_fe_layers(self, architecture: Sequence[tuple]) -> tuple[list, int, int]:
        fe_layers = []
        dim_out = self.image_size
        input_channels = self.input_channels

        for layer_cfg in architecture:
            match layer_cfg:
                case ('Conv2d', num_filters, kernel_size, stride, padding):
                    # Compute the output dimension of the layer
                    dim_out = ((dim_out + 2 * padding - kernel_size) // stride) + 1
                    # Instantiate a convolutional local-loss block
                    fe_layers.append(
                        LocalConv2dBlock(
                            in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size,
                            num_classes=self.num_classes, dim_out=dim_out, optimizer=self.optimizer,
                            non_linearity=self.non_linearity, stride=stride, padding=padding,
                            pred_decoder_dim=self.pred_decoder_dim, fwd_decay_inv=self.fwd_decay_inv,
                            subnet_decay_inv=self.subnet_decay_inv, debug=self.debug,
                            subnet_pooling_type=self.subnet_pooling_type, bias=self.bias, dtype=self.dtype,
                            init=self.init, name_index=len(fe_layers), trainable=self.trainable,
                            local_loss=self.local_loss, beta=self.beta,
                            name=f'conv_block_{len(fe_layers):02d}', device=self.device,
                            subnet_trainable=self.subnet_trainable, lr_amp_factor=self.lr_amp_factor
                        )
                    )
                    input_channels = num_filters
                    # Keep track of a reference to the block
                    self.blocks.append(fe_layers[-1])

                    # Instantiate a Dropout layer if needed
                    if self.fe_dropout_rate != 0.0:
                        fe_layers.append(
                            Dropout(dropout_rate=self.fe_dropout_rate, name=f'dropout_{len(fe_layers):02d}')
                        )

                case ('MaxPool2d', kernel_size, stride):
                    fe_layers.append(
                        MaxPool2d(kernel_size=kernel_size, stride=stride, dtype=self.dtype,
                                  name=f'max_pool_{len(fe_layers):02d}')
                    )
                    dim_out = dim_out // stride

                case ('AvgPool2d', kernel_size, stride):
                    fe_layers.append(
                        IntegerAvgPool2d(kernel_size=kernel_size, stride=stride, name=f'avg_pool_{len(fe_layers):02d}')
                    )
                    dim_out = dim_out // stride

                case ('Dropout', dropout_rate):
                    fe_layers.append(
                        Dropout(dropout_rate=dropout_rate, name=f'dropout_{len(fe_layers):02d}')
                    )

                case ('GlobalMaxPool2d'):
                    fe_layers.append(
                        GlobalMaxPool2d(name=f'global_max_pool_{len(fe_layers):02d}')
                    )
                    dim_out = 1

        return fe_layers, dim_out, input_channels
