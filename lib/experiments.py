from typing import Literal

import matplotlib.pyplot as plt
from tqdm.auto import trange
import seaborn as sns
import numpy as np
import cupy as cp

from lib.utils.data import shuffle_dataset, AugmentationPipeline, RandomCrop, RandomHorizontalFlip
from lib.models.local_loss_models import IntegerLocalLossMLP, IntegerLocalLossCNN
from lib.utils.nn import accuracy, plot_history, EarlyStopping, ReduceLROnPlateau
from lib.utils.enums import NonLinearity, Initialization, OptimizerName, Dataset
from lib.utils.train import train_step_local, val_step
from lib.optim.lr_schedulers import ConstantLR
from lib.utils.misc import to_cpu

sns.set_theme()


def build_MLP(exp_config, X_tr, y_tr) -> IntegerLocalLossMLP:
    return IntegerLocalLossMLP(
        num_fc_layers=exp_config['num_fc_layers'], num_fc_hidden=exp_config['num_fc_hidden'],
        num_classes=y_tr.shape[1], input_dim=X_tr.shape[1] * X_tr.shape[2] * X_tr.shape[3],
        dropout_rate=exp_config['dropout_rate'],
        non_linearity=NonLinearity[exp_config['non_linearity']],
        local_loss=exp_config['local_loss'], optimizer=OptimizerName[exp_config['optimizer']],
        dtype=exp_config['dtype'], debug=exp_config['debug'], bias=exp_config['bias'],
        fwd_decay_inv=exp_config['fwd_decay_inv'],
        subnet_decay_inv=exp_config['subnet_decay_inv'], device=exp_config['device'],
        lr_amp_factor=exp_config['lr_amp_factor'], init=Initialization[exp_config['init']]
    )


def build_CNN(exp_config, X_tr, y_tr) -> IntegerLocalLossCNN:
    return IntegerLocalLossCNN(
        input_channels=X_tr.shape[1], image_size=X_tr.shape[2],
        fe_architecture=exp_config['architecture'],
        optimizer=OptimizerName[exp_config['optimizer']],
        num_fc_layers=exp_config['num_fc_layers'], num_hidden_fc=exp_config['num_fc_hidden'],
        num_classes=y_tr.shape[1], non_linearity=NonLinearity[exp_config['non_linearity']],
        local_loss=exp_config['local_loss'], subnet_pooling_type=exp_config['pooling_type'],
        fe_dropout_rate=exp_config['fe_dropout_rate'], fc_dropout_rate=exp_config['fc_dropout_rate'],
        pred_decoder_dim=exp_config['pred_decoder_dim'], debug=exp_config['debug'],
        bias=exp_config['bias'], dtype=exp_config['dtype'],
        subnet_decay_inv=exp_config['subnet_decay_inv'],
        fwd_decay_inv=exp_config['fwd_decay_inv'], device=exp_config['device'],
        lr_amp_factor=exp_config['lr_amp_factor'], init=Initialization[exp_config['init']]
    )


def run_multiple_experiments_mlp(init_seed, num_experiments, exp_config, data,
                                 show_run_results=False, show_progress_bar: bool = True):
    return run_multiple_experiments('MLP', init_seed, num_experiments, exp_config, data,
                                    show_run_results, show_progress_bar)


def run_multiple_experiments_cnn(init_seed, num_experiments, exp_config, data, show_run_results=False,
                                 show_progress_bar: bool = True):
    return run_multiple_experiments('CNN', init_seed, num_experiments, exp_config, data,
                                    show_run_results, show_progress_bar)


def run_multiple_experiments(model_type, init_seed, num_experiments, exp_config, data,
                             show_run_results=False, show_progress_bar: bool = True):
    models = []
    histories = []
    test_accuracies = []
    local_histories = []

    # Run multiple experiments
    first_iteration = True
    for i in range(num_experiments):
        current_seed = init_seed + i
        m, history, test_acc, local_history = run_experiment(
            model_type,
            exp_config,
            *data,
            current_seed,
            first_run=first_iteration,
            show_run_results=show_run_results,
            show_progress_bar=show_progress_bar
        )

        models.append(m)
        histories.append(history)
        test_accuracies.append(test_acc)
        local_histories.append(local_history)
        first_iteration = False

    # Convert to NumPy arrays
    models = np.array(models)
    histories = np.array(histories)
    test_accuracies = np.array(test_accuracies)
    local_histories = np.array(local_histories)

    return models, histories, test_accuracies, local_histories


def run_experiment(model_type: Literal['MLP', 'CNN'], exp_config, X_tr, X_va, X_te, y_tr, y_va, y_te, seed,
                   show_run_results: bool = False, show_progress_bar: bool = True, first_run: bool = False):
    np.random.seed(seed)
    cp.random.seed(seed)

    # Select the correct model type
    if model_type == 'MLP':
        model = build_MLP(exp_config, X_tr, y_tr)
    elif model_type == 'CNN':
        model = build_CNN(exp_config, X_tr, y_tr)
    else:
        raise ValueError(f'Invalid model type: {model_type}')

    exp_config['model_type'] = model.__class__.__name__
    if first_run:
        print(model.extra_repr(1))

    # Train
    lr_scheduler = ConstantLR(exp_config['lr_inv'])
    n_train_batches = len(X_tr) // exp_config['batch_size']
    n_val_batches = len(X_va) // exp_config['batch_size']
    last_epoch = 0

    xp = cp.get_array_module(X_tr)
    loss_history = xp.empty(exp_config['num_epochs'])
    acc_history = xp.empty(exp_config['num_epochs'])
    val_loss_history = xp.empty(exp_config['num_epochs'])
    val_acc_history = xp.empty(exp_config['num_epochs'])
    local_losses_history = list()
    local_accuracies_history = list()

    for epoch in trange(exp_config['num_epochs'], disable=(not show_progress_bar)):
        # Train and validation loops
        lr_scheduler.step()
        X_tr, y_tr = shuffle_dataset(X_tr, y_tr)

        loss_history[epoch], acc_history[epoch], local_losses, local_accuracies = train_step_local(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            batch_size=exp_config['batch_size'],
            n_train_batches=n_train_batches,
            lr_inv=lr_scheduler.get_lr()
        )

        val_loss_history[epoch], val_acc_history[epoch] = val_step(
            model=model,
            X_val=X_va,
            y_val=y_va,
            batch_size=exp_config['batch_size'],
            n_val_batches=n_val_batches
        )

        local_losses_history.append(local_losses)
        local_accuracies_history.append(local_accuracies)
        last_epoch = epoch

    # Test evaluation
    y_pred = model.predict(X_te, progress_bar=False)
    test_accuracy = float(accuracy(y_te, y_pred))

    # Plot the history of the loss and of the accuracy
    loss_history, val_loss_history = loss_history[:last_epoch + 1], val_loss_history[:last_epoch + 1]
    acc_history, val_acc_history = acc_history[:last_epoch + 1], val_acc_history[:last_epoch + 1]

    if exp_config['device'] == 'cuda':
        # Move training statistics to the CPU
        loss_history, val_loss_history = to_cpu(loss_history), to_cpu(val_loss_history)
        acc_history, val_acc_history = to_cpu(acc_history), to_cpu(val_acc_history)

    histories_list = [loss_history, val_loss_history, acc_history, val_acc_history]
    local_histories_list = [local_losses_history, local_accuracies_history]

    if show_run_results:
        print(f'Final train accuracy: {acc_history[-1]:.2%}')
        print(f'Final validation accuracy: {val_acc_history[-1]:.2%}')
        print(f'Test accuracy: {test_accuracy:.2%}')
        plot_history(loss_history, val_loss_history, acc_history, val_acc_history, log_scale=True)

    return model, histories_list, test_accuracy, local_histories_list


def train_test_experiment(model, exp_config, X_tr, X_te, y_tr, y_te, augmentation_pipeline: AugmentationPipeline = None,
                          show_progress_bar: bool = True, show_local_accuracies: bool = False) -> tuple[float, float]:
    lr_scheduler = ConstantLR(exp_config['lr_inv'])
    early_stopping = EarlyStopping(min_delta=0.01, patience=35, from_epoch=30)
    reduce_lr_plateau = ReduceLROnPlateau(factor=3, min_delta=0.01, patience=15, from_epoch=10)
    n_train_batches = len(X_tr) // exp_config['batch_size']
    n_test_batches = len(X_te) // exp_config['batch_size']
    max_train_acc = 0.0
    max_test_acc = 0.0

    for epoch in trange(exp_config['num_epochs'], disable=(not show_progress_bar)):
        # Train and validation loops
        lr_scheduler.step()
        X_tr, y_tr = shuffle_dataset(X_tr, y_tr)

        loss, acc, local_losses, local_accuracies = train_step_local(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            batch_size=exp_config['batch_size'],
            n_train_batches=n_train_batches,
            augmentation_pipeline=augmentation_pipeline,
            lr_inv=lr_scheduler.get_lr()
        )

        test_loss, test_acc = val_step(
            model=model,
            X_val=X_te,
            y_val=y_te,
            batch_size=exp_config['batch_size'],
            n_val_batches=n_test_batches
        )

        # Keep track of the maximum test accuracy and corresponding train accuracy
        if test_acc > max_test_acc:
            max_train_acc = acc
            max_test_acc = test_acc

        if show_local_accuracies:
            if epoch % 1 == 0:
                print(f"Epoch {epoch}:")
                for i, (local_loss, local_acc) in enumerate(zip(local_losses, local_accuracies)):
                    print(f"- Local Accuracy {i}: {local_acc:.2%}")
                print(f"- Output Accuracy: {acc:.2%}")
                print(f"- Output Val Accuracy: {test_acc:.2%}")
                print("\n")
        else:
            print(f'Epoch {(epoch + 1):>3}/{exp_config["num_epochs"]:<3} '
                  f'- Train accuracy: {acc:07.3%} - Test accuracy: {test_acc:07.3%}')

        # Early stopping
        if early_stopping.early_stop(float(test_acc)):
            print(f"{'-' * 64}\nEarly stopping at epoch {epoch + 1}\n{'-' * 64}")
            break

        # ReduceLROnPlateau
        if reduce_lr_plateau.reduce_lr(float(test_acc)):
            lr_scheduler.set_lr(lr_scheduler.get_lr() * reduce_lr_plateau.factor)
            print(f"{'-' * 64}\nReducing learning rate on plateau at epoch {epoch + 1}\n{'-' * 64}")

    return max_train_acc, max_test_acc


def plot_with_std(ax, values, label, log_scale=False) -> None:
    """ Line plot of the mean with 95% confidence interval """
    values_mean = np.mean(values, axis=0)
    values_std = np.std(values, axis=0)
    if log_scale:
        ax.semilogy(values_mean, label=label)
    else:
        ax.plot(values_mean, label=label)
    ax.fill_between(np.arange(len(values_mean)), values_mean - (2 * values_std), values_mean + (2 * values_std),
                    alpha=0.25)
    ax.grid(True)
    ax.legend()


def plot_avg_history(histories: np.ndarray, figsize: tuple[int, int] = (16, 5), log_scale: bool = False) -> None:
    loss_history, val_loss_history = histories[:, 0, :], histories[:, 1, :]
    acc_history, val_acc_history = histories[:, 2, :], histories[:, 3, :]
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

    plot_with_std(axs[0], loss_history, label='Train', log_scale=log_scale)
    plot_with_std(axs[0], val_loss_history, label='Validation', log_scale=log_scale)
    axs[0].set_title('Loss')
    axs[0].grid(True)
    axs[0].legend()

    plot_with_std(axs[1], acc_history, label='Train')
    plot_with_std(axs[1], val_acc_history, label='Validation')
    axs[1].set_title('Accuracy')
    axs[1].grid(True)
    axs[1].legend()


def plot_weights_norms(models: list, figsize=(12, 6)) -> None:
    # Extract the norms of the weights of each layer
    weight_norms = []
    for model in models:
        layers = model.get_layers_with_parameters()
        norms = []
        for layer in layers:
            # Compute the mean absolute value of the weights
            mean_value = np.mean(np.abs(layer.weights))
            norms.append(to_cpu(mean_value))
        weight_norms.append(norms)
    weight_norms = np.array(weight_norms)

    # Bar plot of the norms with seaborn
    _, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(weight_norms, capsize=.25, estimator='mean', errorbar=('ci', 95), ax=ax, palette='deep')
    ax.set_title("Mean absolute value of the weights of each layer")
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean absolute value')


def print_accuracies(histories: np.ndarray, test_accuracies: np.ndarray, local_histories: np.ndarray) -> None:
    """ Print the average train/val/test accuracies with 95% confidence interval """
    avg_train_accuracy = np.mean(histories[:, 2, -1]) * 100
    std_train_accuracy = np.std(histories[:, 2, -1]) * 100
    avg_val_accuracy = np.mean(histories[:, 3, -1]) * 100
    std_val_accuracy = np.std(histories[:, 3, -1]) * 100
    avg_test_accuracy = np.mean(test_accuracies) * 100
    std_test_accuracy = np.std(test_accuracies) * 100

    # Accuracies of local loss blocks
    local_accuracies = local_histories[:, 1, -1, :]
    avg_local_accuracies = np.mean(local_accuracies, axis=0) * 100
    std_local_accuracies = np.std(local_accuracies, axis=0) * 100
    for i in range(len(avg_local_accuracies)):
        avg_local_acc = float(avg_local_accuracies[i])
        std_local_acc = float(std_local_accuracies[i])
        print(f'Block {i} train accuracy: ({avg_local_acc:.3f} ± {std_local_acc * 2:.3f})%')

    # Accuracies of the last layer
    print(f'Average train accuracy: ({avg_train_accuracy:.3f} ± {std_train_accuracy * 2:.3f})%')
    print(f'Average validation accuracy: ({avg_val_accuracy:.3f} ± {std_val_accuracy * 2:.3f})%')
    print(f'Average test accuracy: ({avg_test_accuracy:.3f} ± {std_test_accuracy * 2:.3f})%')


def adjust_hyperparameters(exp_config) -> None:
    # Adjust init according to non-linearity
    match exp_config['non_linearity']:
        case NonLinearity.POCKET_LEAKY_RELU.name:
            exp_config['init'] = Initialization.UNIFORM_KAIMING_LEAKY_RELU.name
        case NonLinearity.NITRO_LEAKY_RELU.name:
            exp_config['init'] = Initialization.UNIFORM_KAIMING_LEAKY_RELU.name
        case NonLinearity.POCKET_BIPOLAR_LEAKY_RELU.name:
            exp_config['init'] = Initialization.UNIFORM_STD.name
        case NonLinearity.POCKET_RELU.name:
            exp_config['init'] = Initialization.UNIFORM_KAIMING_RELU.name
        case NonLinearity.POCKET_BIPOLAR_RELU.name:
            exp_config['init'] = Initialization.UNIFORM_STD.name
        case NonLinearity.POCKET_TANH.name:
            exp_config['init'] = Initialization.UNIFORM_XAVIER.name

    # Adjust architecture according to pooling type
    if exp_config['pooling_type'] is not None:
        pooling_layer = 'MaxPool2d' if exp_config['pooling_type'] == 'max' else 'AvgPool2d'
        for layer in exp_config['architecture']:
            if layer[0] in ['MaxPool2d', 'AvgPool2d']:
                layer[0] = pooling_layer


def build_augmentation_pipeline(exp_config) -> AugmentationPipeline | None:
    if exp_config['data_augmentation']:
        match Dataset[exp_config['dataset']]:
            case Dataset.MNIST:
                return AugmentationPipeline([
                    RandomCrop(28, padding=2)
                ])
            case Dataset.FASHION_MNIST:
                return AugmentationPipeline([
                    RandomCrop(28, padding=2),
                    RandomHorizontalFlip()
                ])
            case Dataset.CIFAR10:
                return AugmentationPipeline([
                    RandomCrop(32, padding=4),
                    RandomHorizontalFlip()
                ])
            case Dataset.CIFAR100:
                return AugmentationPipeline([
                    RandomCrop(32, padding=4),
                    RandomHorizontalFlip()
                ])
            case _:
                raise ValueError(f'Augmentation not implemented for dataset: {Dataset[exp_config["dataset"]]}')
    else:
        return None
