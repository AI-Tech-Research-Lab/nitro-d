from typing import Literal

import numpy as np

from lib.blocks.LocalLossBlock import LocalLossBlock
from lib.layers.conv import IntegerConv2d
from lib.layers.linear import IntegerLinear
from lib.layers.misc import Flatten, ConstantScaling
from lib.layers.pooling import IntegerAvgPool2d, MaxPool2d
from lib.models.sequential import Sequential
from lib.utils.enums import NonLinearity, Initialization, OptimizerName
from lib.utils.nn import create_non_linearity


class LocalConv2dBlock(LocalLossBlock):
    """
    A local-loss block containing the following layers:
    - Integer Conv2d layer
    - Scaling layer
    - Non-linearity

    Attributes
    _________
    in_channels: int
        Number of input channels of the block
    out_channels: int
        Number of output channels of the block
    kernel_size: int
        Kernel size of the convolutional layer
    num_classes: int
        Number of classes of the classification task, used in the local prediction loss
    dim_out: int
        Output dimension of the convolutional layer.
        It is used to automatically compute the size of the linear subnetwork's decoder layer.
    non_linearity: Enum
        Non-linear activation function to be used in the block
    lr_amp_factor: int, default=None
        Amplification factor of the learning rate to be used in the forward part of the block.
        If None, it is computed automatically.
    stride: int, default=1
        Stride of the convolutional layer
    padding: int, default=1
        Padding of the convolutional layer
    pred_decoder_dim: int, default=2048
        Dimension of the linear subnetwork's decoder layer.
        It can be chosen arbitrarily and a pooling layer will be added to the linear subnetwork if needed.
        The larger the dimension, the more accurate the prediction loss will be (aside from overfitting).
    fwd_decay_inv: int, default=0
        Inverse of the decay rate to be used in the Integer Conv2d layer
    subnet_decay_inv: int, default=0
        Inverse of the decay rate to be used in the Integer Linear layer
    bias: bool, default=True
        Whether to use a bias in the Integer layers
    layers: Sequential
        The forward layers of the block
    last_activation: np.ndarray
        The output of the block at the previous training step, used to compute the local loss
    pred_loss_net: Sequential
        The learning layers used to compute the local loss
    subnet_pooling_type: str, default='max'
        Type of pooling layer to be used in the linear subnetwork
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_classes: int, dim_out: int,
                 optimizer: OptimizerName, non_linearity: NonLinearity, stride: int = 1, padding: int = 1,
                 pred_decoder_dim: int = 2048, fwd_decay_inv: int = 0, subnet_decay_inv: int = 0, debug: bool = False,
                 subnet_pooling_type: Literal['avg', 'max'] = 'max', bias: bool = True, dtype: str = 'int32',
                 init: Initialization = Initialization.UNIFORM_STD, name_index: int = 0, trainable: bool = True,
                 local_loss: Literal['pred', 'sim', 'predsim'] = 'pred', beta: float = 0.99, name: str = None,
                 device: Literal['cpu', 'cuda'] = 'cpu', subnet_trainable: bool = True,
                 lr_amp_factor: int = None) -> None:
        super().__init__(local_loss, beta, subnet_trainable,
                         debug=debug, device=device, trainable=trainable, name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.dim_out = dim_out
        self.non_linearity = non_linearity
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pred_decoder_dim = pred_decoder_dim
        self.subnet_decay_inv = subnet_decay_inv
        self.fwd_decay_in = fwd_decay_inv
        self.name_index = name_index
        self.optimizer = optimizer
        self.subnet_pooling_type = subnet_pooling_type

        # Compute the suggested amplification factor of the learning rate for the first layer
        self.lr_amp_factor = (2 ** 6) * num_classes if lr_amp_factor is None else lr_amp_factor

        # Instantiate the forward layers
        fwd_conv = IntegerConv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias, propagate_backward=False,
            name=f'conv_encoder_{name_index:02d}', dtype=dtype, optimizer=optimizer, debug=debug,
            decay_rate_inv=fwd_decay_inv, device=device, trainable=trainable,
            lr_amp_factor=self.lr_amp_factor, init=init
        )
        layers = [
            fwd_conv,
            ConstantScaling(fwd_factor=(2 ** 8) * (kernel_size ** 2) * in_channels, name=f'scaling_{name_index:02d}'),
            create_non_linearity(non_linearity, name=f'non_linearity_{name_index:02d}'),
        ]

        # Instantiate the learning layers
        ks_h, ks_w = 1, 1
        match self.local_loss:
            case 'pred':
                subnet_layers = []
                # Resolve average-pooling kernel size in order for flattened dim to match pred_encoder_dim
                dim_out_h, dim_out_w = dim_out, dim_out
                dim_in_decoder = out_channels * dim_out_h * dim_out_w

                while dim_in_decoder > pred_decoder_dim and ks_h < dim_out:
                    ks_h *= 2
                    dim_out_h = np.ceil(dim_out / ks_h).astype(dtype)
                    dim_in_decoder = out_channels * dim_out_h * dim_out_w

                    if dim_in_decoder > pred_decoder_dim:
                        ks_w *= 2
                        dim_out_w = np.ceil(dim_out / ks_w).astype(dtype)
                        dim_in_decoder = out_channels * dim_out_h * dim_out_w

                if ks_h > 1 or ks_w > 1:
                    # Size is not correct, need to perform pooling
                    pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                    pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2

                    if subnet_pooling_type == 'max':
                        subnet_layers.append(
                            MaxPool2d(
                                kernel_size=(ks_h, ks_w),
                                stride=(ks_h, ks_w),
                                padding=(pad_h, pad_w),
                                name=f'pooling_pred_{self.name_index:02d}'
                            )
                        )
                    elif subnet_pooling_type == 'avg':
                        subnet_layers.append(
                            IntegerAvgPool2d(
                                kernel_size=(ks_h, ks_w),
                                stride=(ks_h, ks_w),
                                padding=(pad_h, pad_w),
                                name=f'pooling_pred_{self.name_index:02d}'
                            )
                        )
                    else:
                        raise ValueError(f'Invalid pooling type: {subnet_pooling_type}')

                subnet_layers.append(Flatten(name=f'flatten_pred_{self.name_index:02d}'))
                pred_linear = IntegerLinear(
                    in_features=dim_in_decoder, out_features=num_classes, bias=bias,
                    decay_rate_inv=subnet_decay_inv, name=f'linear_pred_{self.name_index:02d}',
                    dtype=dtype, debug=debug, init=init, optimizer=optimizer, device=device,
                    trainable=subnet_trainable
                )
                subnet_layers.append(pred_linear)
                subnet_layers.append(
                    ConstantScaling(
                        fwd_factor=(2 ** 8) * dim_in_decoder,
                        name=f'scaling_pred_{self.name_index:02d}'
                    )
                )
                self.pred_loss_net = Sequential(subnet_layers, debug=debug, device=device, trainable=trainable)

            case 'sim':
                ...
            case 'predsim':
                ...

        self.ks_h, self.ks_w = ks_h, ks_w
        self.layers = Sequential(layers, debug=debug, device=device, trainable=trainable)
