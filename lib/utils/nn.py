import warnings

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from lib.layers.activations import PocketReLU, PocketLeakyReLU, PocketTanh
from lib.layers.activations import NitroLeakyReLU, BipolarLeakyReLU, BipolarPocketReLU
from lib.utils.enums import NonLinearity, OptimizerName
from lib.utils.misc import to_cpu, truncated_division
from lib.optim.optimizers import IntegerSGD
from lib.optim.Optimizer import Optimizer
from lib.layers.modules import Module

sns.set_theme()


def l2_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ Compute the RSS loss function """
    return np.sum((np.subtract(y_true, y_pred)) ** 2)


def l2_loss_grad(y_true: np.ndarray, y_pred: np.ndarray, dtype: str = 'int8') -> np.ndarray:
    """ Computes the gradient of the RSS loss function """
    return np.subtract(y_pred, y_true, dtype=dtype)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """ Compute the MSE loss function """
    return l2_loss(y_true, y_pred) / len(y_true)


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray, dtype: str = 'int8') -> np.ndarray:
    """ Computes the gradient of the MSE loss function using integer-only arithmetic """
    return truncated_division(l2_loss_grad(y_true, y_pred, dtype=dtype), len(y_true))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.equal(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).sum() / len(y_true)


def create_optimizer(optimizer: OptimizerName, layer: Module, weight_decay_inv: int = 0) -> Optimizer:
    match optimizer:
        case OptimizerName.INTEGER_SGD:
            return IntegerSGD(layer, weight_decay_inv=weight_decay_inv)
        case _:
            raise ValueError(f'Invalid optimizer: {optimizer}')


def create_non_linearity(non_linearity: NonLinearity, **kwargs) -> Module:
    match non_linearity:
        case NonLinearity.POCKET_TANH:
            return PocketTanh(**kwargs)
        case NonLinearity.POCKET_RELU:
            return PocketReLU(**kwargs)
        case NonLinearity.POCKET_BIPOLAR_RELU:
            return BipolarPocketReLU(**kwargs)
        case NonLinearity.POCKET_LEAKY_RELU:
            return PocketLeakyReLU(**kwargs)
        case NonLinearity.NITRO_LEAKY_RELU:
            return NitroLeakyReLU(**kwargs)
        case NonLinearity.POCKET_BIPOLAR_LEAKY_RELU:
            return BipolarLeakyReLU(**kwargs)
        case _:
            raise ValueError(f'Invalid non-linearity: {non_linearity}')


def plot_history(loss_history: list[float], val_loss_history: list[float],
                 acc_history: list[float], val_acc_history: list[float],
                 figsize: tuple[int, int] = (16, 5), log_scale: bool = False) -> None:
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_title('Loss')
    if log_scale:
        axs[0].semilogy(loss_history, label='Train')
        axs[0].semilogy(val_loss_history, label='Validation')
    else:
        axs[0].plot(loss_history, label='Train')
        axs[0].plot(val_loss_history, label='Validation')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title('Accuracy')
    axs[1].plot(acc_history, label='Train')
    axs[1].plot(val_acc_history, label='Validation')
    axs[1].grid(True)
    axs[1].legend()
    plt.show()


def plot_mean_weights(model, figsize: tuple[int, int] = (12, 6)) -> None:
    # Extract the norms of the weights of each layer
    weight_norms = []
    for layer in model.get_layers_with_parameters():
        # Compute the mean absolute value of the weights
        mean_value = np.mean(np.abs(layer.weights))
        weight_norms.append(to_cpu(mean_value))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Bar plot of the norms with seaborn
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.barplot(np.array(weight_norms), ax=ax, palette='deep')
        ax.set_title("Mean absolute value of the weights of each layer")
        ax.set_xlabel('Layer index')
        ax.set_ylabel('Mean absolute value')


def plot_filters_norm(model, figsize: tuple[int, int] = (12, 16)) -> None:
    # Extract the norms of the filters of each convolutional layer
    filters_norms = []
    conv_layers = model.get_conv_layers()
    for layer in conv_layers:
        filters = layer.weights.reshape(layer.weights.shape[0], -1)
        filter_norms = np.linalg.norm(filters, axis=1, ord=1)
        # Normalize w.r.t. the number of elements used to compute the norm
        filter_norms /= filters.shape[1]
        filters_norms.append(to_cpu(filter_norms))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Generate one bar plot for each layer
        _, axs = plt.subplots(len(filters_norms), 1, figsize=figsize)
        for i, ax in enumerate(np.array(axs).flatten()):
            # Bar plot of the norms with seaborn
            sns.barplot(filters_norms[i], capsize=.25, estimator='mean', errorbar=('ci', 95), ax=ax, palette='deep')
            ax.set_title(f"Normalized L1 norm of the filters of conv layer {i}")
            ax.set_xlabel('Filter index')
            ax.set_ylabel('Normalized L1 norm')
            ax.grid(True)


class EarlyStopping:
    def __init__(self, min_delta: float = 0.01, patience: int = 10, from_epoch: int = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.from_epoch = from_epoch
        self.max_val_acc = 0
        self.counter = 0
        self.epoch = 0

    def early_stop(self, val_acc: float) -> bool:
        self.epoch += 1
        if self.epoch < self.from_epoch:
            return False

        if val_acc < (self.max_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Early stopping condition met
                return True

        else:
            # Reset the counter and update the maximum validation accuracy
            self.max_val_acc = val_acc
            self.counter = 0
            return False


class ReduceLROnPlateau():
    def __init__(self, factor: int = 2, min_delta: float = 0.01, patience: int = 10, from_epoch: int = 0) -> None:
        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.from_epoch = from_epoch
        self.max_val_acc = 0.0
        self.counter = 0
        self.epoch = 0

    def reduce_lr(self, val_acc: float) -> bool:
        self.epoch += 1
        if self.epoch < self.from_epoch:
            return False

        if val_acc < (self.max_val_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                # Plateau detected: reduce the learning rate
                self.counter = 0
                return True

        else:
            # Reset the counter and update the maximum validation accuracy
            self.max_val_acc = val_acc
            self.counter = 0
            return False
