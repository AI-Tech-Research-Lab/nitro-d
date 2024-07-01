from typing import Literal

from lib.blocks.LocalLossBlock import LocalLossBlock
from lib.layers.linear import IntegerLinear
from lib.layers.misc import ConstantScaling
from lib.models.sequential import Sequential
from lib.utils.enums import NonLinearity, Initialization, OptimizerName
from lib.utils.nn import create_non_linearity


class LocalLinearBlock(LocalLossBlock):
    """
    A local-loss block containing the following layers:
    - Integer Linear layer
    - Scaling layer
    - Non-linearity
    - Dropout layer (optional)

    Attributes
    _________
    in_features: int
        Number of input features of the block
    out_features: int
        Number of output features of the block
    num_classes: int
        Number of classes of the classification task, used in the local prediction loss
    non_linearity: Enum
        Non-linear activation function to be used in the block
    lr_amp_factor: int, default=None
        Amplification factor of the learning rate to be used in the forward part of the block.
        If None, it is computed automatically.
    fwd_decay_inv: int, default=0
        Inverse of the decay rate to be used in the Integer Linear forward layer
    subnet_decay_inv: int, default=0
        Inverse of the decay rate to be used in the Integer Linear learning layer
    bias: bool
        Whether to use a bias in the Integer layers
    layers: Sequential
        The forward layers of the block
    last_activation: np.ndarray
        The output of the block at the previous training step, used to compute the local loss
    pred_loss_net: Sequential
        The learning layers used to compute the local loss
    """

    def __init__(self, in_features: int, out_features: int, num_classes: int, non_linearity: NonLinearity,
                 optimizer: OptimizerName, name: str = None, fwd_decay_inv: int = 0, subnet_decay_inv: int = 0,
                 debug: bool = False, bias: bool = True, dtype: str = 'int32', name_index: int = 0,
                 init: Initialization = Initialization.UNIFORM_STD, lr_amp_factor: int = None, beta: float = 0.99,
                 trainable: bool = True, local_loss: Literal['pred', 'sim', 'predsim'] = 'pred',
                 device: Literal['cpu', 'cuda'] = 'cpu', subnet_trainable: bool = True) -> None:
        super().__init__(local_loss, beta, subnet_trainable,
                         debug=debug, device=device, trainable=trainable, name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.non_linearity = non_linearity
        self.fwd_decay_inv = fwd_decay_inv
        self.subnet_decay_inv = subnet_decay_inv
        self.optimizer = optimizer

        # Compute the suggested amplification factor of the learning rate for the first layer
        self.lr_amp_factor = (2 ** 6) * num_classes if lr_amp_factor is None else lr_amp_factor

        # Instantiate the forward layers
        fwd_linear = IntegerLinear(
            in_features=in_features, out_features=out_features, bias=bias,
            decay_rate_inv=fwd_decay_inv, name=f'linear_encoder_{name_index:02d}', dtype=dtype,
            propagate_backward=False, debug=debug, init=init, optimizer=optimizer,
            lr_amp_factor=self.lr_amp_factor, device=device, trainable=trainable
        )
        layers = [
            fwd_linear,
            ConstantScaling(fwd_factor=(2 ** 8) * in_features, name=f'scaling_{name_index:02d}'),
            create_non_linearity(non_linearity, name=f'non_linearity_{name_index:02d}')
        ]

        self.layers = Sequential(layers, debug=debug, device=device, trainable=trainable)

        # Build the local loss subnetwork
        match self.local_loss:
            case 'pred':
                pred_linear = IntegerLinear(
                    in_features=out_features, out_features=num_classes, bias=bias,
                    decay_rate_inv=subnet_decay_inv, name=f'linear_pred_{name_index:02d}',
                    dtype=dtype, debug=debug, init=init, optimizer=optimizer, device=device,
                    trainable=subnet_trainable
                )
                subnet_layers = [
                    pred_linear,
                    ConstantScaling(fwd_factor=(2 ** 8) * out_features, name=f'scaling_pred_{name_index:02d}')
                ]
                self.pred_loss_net = Sequential(subnet_layers, debug=debug, device=device, trainable=trainable)

            case 'sim':
                ...
            case 'predsim':
                ...
