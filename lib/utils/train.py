import numpy as np

from lib.models.models import Model, LocalLossModel
from lib.utils.data import AugmentationPipeline
from lib.utils.nn import accuracy, l2_loss_grad, l2_loss


def train_step_local(model: LocalLossModel, X_train: np.ndarray, y_train: np.ndarray, batch_size: int,
                     n_train_batches: int, augmentation_pipeline: AugmentationPipeline = None,
                     **bw_args) -> tuple[float, float, np.ndarray, np.ndarray]:
    loss, acc = 0, 0
    model.train()
    blocks_loss = np.zeros(len(model.blocks))
    blocks_acc = np.zeros(len(model.blocks))

    for batch_id in range(n_train_batches):
        batch_start = batch_id * batch_size
        batch_end = batch_start + batch_size
        x = X_train[batch_start:batch_end]
        y = y_train[batch_start:batch_end]

        if augmentation_pipeline is not None:
            x = augmentation_pipeline(x)

        y_pred = model.forward(x)
        acc += accuracy(y, y_pred)
        loss += l2_loss(y, y_pred)

        for i in range(len(model.blocks)):
            # Compute the local loss and the accuracy of early exit classifiers
            blocks_loss[i] += model.blocks[i].loss
            blocks_acc[i] += model.blocks[i].train_accuracy

        l2_grad = l2_loss_grad(y, y_pred)
        model.backward(l2_grad, y_true=y, **bw_args)

    return loss / n_train_batches, acc / n_train_batches, blocks_loss / n_train_batches, blocks_acc / n_train_batches


def val_step(model: Model, X_val: np.ndarray, y_val: np.ndarray,
             batch_size: int, n_val_batches: int) -> tuple[float, float]:
    val_loss, val_acc = 0, 0
    model.eval()

    for batch_id in range(n_val_batches):
        batch_start = batch_id * batch_size
        batch_end = batch_start + batch_size
        x = X_val[batch_start:batch_end]
        y = y_val[batch_start:batch_end]

        y_pred = model.forward(x)
        val_acc += accuracy(y, y_pred)
        val_loss += l2_loss(y, y_pred)

    return val_loss / n_val_batches, val_acc / n_val_batches
