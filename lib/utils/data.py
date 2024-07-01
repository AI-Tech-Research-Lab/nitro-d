from abc import ABC, abstractmethod
from types import SimpleNamespace

import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets

from lib.utils.enums import Dataset
from lib.utils.misc import int_mean_and_var, int_sqrt_vectorized, to_gpu, int_mean_and_mad, to_cpu, \
    truncated_division


def download_dataset(dataset: Dataset, add_padding: bool = True, data_path: str = '.data') -> \
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
        Helper function which downloads the dataset using torchvision.
        The dataset is pre-processed so that it is compatible with NumPy, and it is of data type uint8.

        Parameters
        ----------
        dataset: Enum
            Enum representing dataset to be downloaded
        add_padding: bool, default=True
            Whether to add padding so that datasets with 28x28 images become 32x32
        data_path: str, default='.data'
            Path to the data folder. If None, .data is used

        Returns
        -------
        train_dataset, test_dataset
            A tuple containing the train dataset and the test dataset, further split into data and labels.
        """
    train_args = dict(root=data_path, train=True, download=True)
    test_args = dict(root=data_path, train=False, download=True)

    def process_grayscale(train_split, test_split) -> tuple[np.ndarray, np.ndarray]:
        train_split = train_split.data.numpy()
        test_split = test_split.data.numpy()
        train_split = np.expand_dims(train_split, axis=1)
        test_split = np.expand_dims(test_split, axis=1)
        pad_width = np.array(((0, 0), (0, 0), (2, 2), (2, 2)))
        if add_padding:
            train_split = np.pad(train_split, pad_width=pad_width, mode='constant', constant_values=0)
            test_split = np.pad(test_split, pad_width=pad_width, mode='constant', constant_values=0)
        return train_split, test_split

    def process_rgb(train_split, test_split) -> tuple[np.ndarray, np.ndarray]:
        train_split = train_split.transpose(0, 3, 1, 2)
        test_split = test_split.transpose(0, 3, 1, 2)
        return train_split, test_split

    def convert_to_grayscale(train_split, test_split) -> tuple[np.ndarray, np.ndarray]:
        train_split = train_split.transpose(0, 3, 1, 2)
        test_split = test_split.transpose(0, 3, 1, 2)
        train_split = np.mean(train_split, axis=1, keepdims=True)
        test_split = np.mean(test_split, axis=1, keepdims=True)
        return train_split, test_split

    match dataset:
        case Dataset.MNIST:
            train_set = datasets.MNIST(**train_args)
            test_set = datasets.MNIST(**test_args)
            train_data, test_data = process_grayscale(train_set.data, test_set.data)

        case Dataset.FASHION_MNIST:
            train_set = datasets.FashionMNIST(**train_args)
            test_set = datasets.FashionMNIST(**test_args)
            train_data, test_data = process_grayscale(train_set.data, test_set.data)

        case Dataset.CIFAR10:
            train_set = datasets.CIFAR10(**train_args)
            test_set = datasets.CIFAR10(**test_args)
            train_data, test_data = process_rgb(train_set.data, test_set.data)

        case Dataset.CIFAR10_GRAY:
            train_set = datasets.CIFAR10(**train_args)
            test_set = datasets.CIFAR10(**test_args)
            train_data, test_data = convert_to_grayscale(train_set.data, test_set.data)

        case Dataset.CIFAR100:
            train_set = datasets.CIFAR100(**train_args)
            test_set = datasets.CIFAR100(**test_args)
            train_data, test_data = process_rgb(train_set.data, test_set.data)

        case Dataset.WHITE_NOISE:
            train_shape, test_shape = (50_000, 1, 32, 32), (10_000, 1, 32, 32)
            train_set, test_set = dict(), dict()
            train_data = np.random.randint(0, 255, size=train_shape)
            train_set["targets"] = np.random.randint(0, 10, size=train_shape[0])
            test_data = np.random.randint(0, 255, size=test_shape)
            test_set["targets"] = np.random.randint(0, 10, size=test_shape[0])
            train_set = SimpleNamespace(**train_set)
            test_set = SimpleNamespace(**test_set)

        case _:
            raise ValueError(f'Invalid dataset: {dataset}')

    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set.targets)

    return ((train_data.astype(np.uint8), train_targets.astype(np.uint8)),
            (test_data.astype(np.uint8), test_targets.astype(np.uint8)))


def shuffle_dataset(X: np.ndarray, y: np.ndarray, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle the dataset using a fixed seed if needed.
    When shuffling the data during preprocessing, it is important to use a fixed seed to ensure reproducibility.
    When shuffling during training, do not provide a seed to ensure that the model sees a different order at each epoch.

    Attributes
    ----------
    X: np.ndarray
        Data to be shuffled
    y: np.ndarray
        Labels to be shuffled
    seed: int, default=None
        Random seed to be used for shuffling

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The shuffled data and labels
    """
    if seed is not None:
        cp.random.seed(seed)
        np.random.seed(seed)

    # Generate and shuffle the set of indices
    idx_list = np.arange(len(X))
    np.random.shuffle(idx_list)

    # Apply the new ordering to the dataset
    X = X[idx_list]
    y = y[idx_list]
    return X, y


def plot_images(x_train: np.ndarray, y_train: np.ndarray, num_img: int = 6, seed: int = None) -> None:
    """
    Display a sample of images from the training-validation dataset.

    Parameters
    ----------
    x_train: np.ndarray
        Training data
    y_train: np.ndarray
        Training labels
    num_img: int, default=6
        Number of images to display
    seed: int, default=None
        Random seed to be used for shuffling

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, num_img, figsize=(20, 20))

    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)

    # Iterate through the selected number of images
    for i in range(num_img):
        sample_id = np.random.randint(0, len(x_train))
        ax = axes[i % num_img]
        image = to_cpu(x_train[sample_id])
        ax.imshow(image.transpose(1, 2, 0), cmap='gray')
        ax.set_title(y_train[sample_id])
        ax.axis('off')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()


def normalize_channels(x_train, x_val, x_test, standardize: bool = True, print_stats: bool = True) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize all the data using statistics computed on the training set.
    If standardize is True, the data is normalized so that it has a zero-mean and a standard deviation of 64.
    With a std=64, 95% of the data will be in the range [-128, 128].
    Otherwise, it is only centered so that it has a zero-mean.
    Mean Absolute Deviation is used in place of Standard Deviation to measure the dispersion around the mean.

    Parameters
    ----------
    x_train: np.ndarray
        Training data
    x_val: np.ndarray
        Validation data
    x_test: np.ndarray
        Test data
    standardize: bool, default=True
        Whether to standardize the data or only center it
    print_stats: bool, default=True
        Whether to print the new statistics after normalization

    Returns
    -------
    x_train, x_val, x_test: tuple[np.ndarray, np.ndarray, np.ndarray]
        The normalized training, validation and test data
    """
    # To avoid overflow, use a subset of the data to compute statistics
    x_sample = x_train[:min(len(x_train), 25_000)]
    mean, mad = int_mean_and_mad(x_sample.astype(np.int32), axis=(0, 2, 3), keepdims=True)

    def normalize(x):
        # Apply standardization
        x_mean = x - mean
        if standardize:
            # mad = std / 1.25 => 64 / 1.25 ~ 51
            x_norm = truncated_division(x_mean * 51, mad)
        else:
            x_norm = x_mean
        # Clip to the int8 range
        return np.clip(x_norm, np.iinfo(np.int8).min, np.iinfo(np.int8).max).astype(np.int8)

    x_train = normalize(x_train)
    if x_val is not None:
        x_val = normalize(x_val)
    x_test = normalize(x_test)

    if print_stats:
        x_sample = x_train[:min(len(x_train), 25_000)]
        new_mean, new_var = int_mean_and_var(x_sample, axis=(0, 2, 3), keepdims=False)
        new_std = int_sqrt_vectorized(new_var)
        print(f'Data normalized. New statistics:\n'
              f'-  Min: {x_train.min(axis=(0, 2, 3)).squeeze()}\n'
              f'-  Max: {x_train.max(axis=(0, 2, 3)).squeeze()}\n'
              f'- Mean: {new_mean.squeeze()}\n'
              f'-  Std: {new_std.squeeze()}\n')

    return x_train, x_val, x_test


def load_integer_dataset(config, ohe_values: tuple[int, int] = (0, 32), add_padding: bool = False,
                         val_dim: int = 10_000, test_dim: int = 10_000, show_images: bool = True,
                         show_log: bool = True, data_path: str = '.data') \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, subsample, normalize and move data to the GPU if required.

    Parameters
    ----------
    config: dict
        A dictionary with the configuration parameters
    ohe_values: tuple[int, int], default=(0, 32)
        Tuple containing the values to be used in the one-hot encoding of the labels
    add_padding: bool, default=False
        Whether to add padding so that datasets with 28x28 images become 32x32
    val_dim: int, default = 10_000
        Desired size of the validation set
    test_dim: int, default= 10_000
        Desired size of the test set
    show_images: bool, default=True
        Whether to display a sample of images from the training dataset
    show_log: bool, default=True
        Whether to print information about the dataset
    data_path: str, default='.data'
        Path to the data folder. If None, .data is used

    Returns
    -------
    x_train, x_val, x_test, y_train, y_val, y_test: tuple[np.ndarray, ...]
        Tuple containing the training, validation and test data and labels
    """
    return _load_dataset(config, ohe_values, add_padding=add_padding, val_dim=val_dim, test_dim=test_dim,
                         show_images=show_images, show_log=show_log, data_path=data_path)


def _load_dataset(config, ohe_values: tuple[int, int] = (0, 32), add_padding: bool = True, val_dim: int = 10_000,
                  test_dim: int = 10_000, show_images: bool = True, show_log: bool = True, data_path: str = '.data') \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load, subsample, normalize and move data to the GPU if required.

    Parameters
    ----------
    config: dict
        A dictionary with the configuration parameters
    ohe_values: tuple[int, int], default=(0, 32)
        Tuple containing the values to be used in the one-hot encoding of the labels
    add_padding: bool, default=True
        Whether to add padding so that datasets with 28x28 images become 32x32
    val_dim: int, default = 10_000
        Desired size of the validation set
    test_dim: int, default= 10_000
        Desired size of the test set
    show_images: bool, default=True
        Whether to display a sample of images from the training dataset
    show_log: bool, default=True
        Whether to print information about the dataset
    data_path: str, default='.data'
        Path to the data folder. If None, .data is used

    Returns
    -------
    x_train, x_val, x_test, y_train, y_val, y_test: tuple[np.ndarray, ...]
        Tuple containing the training, validation and test data and labels
    """
    train, test = download_dataset(Dataset[config['dataset']], add_padding=add_padding, data_path=data_path)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _preprocess_dataset(
        train_set=train,
        test_set=test,
        val_dim=val_dim,
        test_dim=test_dim,
        ohe_values=ohe_values
    )

    # Subsample if required
    num_samples = int(len(x_train) * config['subsample_ratio'])
    x_train = x_train[:num_samples]
    y_train = y_train[:num_samples]

    if config['subsample_ratio'] < 1:
        if show_log:
            print(f'Subsampling applied: training on {len(x_train)} samples')
    else:
        if show_log:
            print(f'Training on {len(x_train)} samples')

    if config['normalize']:
        # Normalize the data if required
        x_train, x_val, x_test = normalize_channels(x_train, x_val, x_test, print_stats=show_log)

    # Cast data and labels to int8
    x_train = x_train.astype(np.int8)
    y_train = y_train.astype(np.int8)
    if x_val is not None:
        x_val = x_val.astype(np.int8)
        y_val = y_val.astype(np.int8)
    x_test = x_test.astype(np.int8)
    y_test = y_test.astype(np.int8)

    if config['device'] == 'cuda':
        # Move data to the GPU
        x_train, y_train = to_gpu(x_train), to_gpu(y_train)
        if x_val is not None:
            x_val, y_val = to_gpu(x_val), to_gpu(y_val)
        x_test, y_test = to_gpu(x_test), to_gpu(y_test)
        if show_log:
            print(f'Moved data to device: {x_train.device}')

    if show_log:
        print(f'Train set:  {x_train.shape}, {y_train.shape}')
        if x_val is not None:
            print(f'Val set:    {x_val.shape}, {y_val.shape}')
        print(f'Test set:   {x_test.shape}, {y_test.shape}')
        print(f'Data types: ({x_train.dtype}, {y_train.dtype})')

    if show_images:
        images = x_train + 128
        plot_images(images, y_train, num_img=6)

    return x_train, x_val, x_test, y_train, y_val, y_test


def _preprocess_dataset(train_set, test_set, val_dim: int = 10_000, test_dim: int = 10_000,
                        ohe_values: tuple[int, int] = (0, 32)) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """
    Prepare a train-validation-test split, scale the data in the correct range and apply one-hot encoding.

    Parameters
    ----------
    train_set: tuple[np.ndarray, np.ndarray]
        Tuple containing the training data and labels
    test_set: tuple[np.ndarray, np.ndarray]
        Tuple containing the test data and labels
    val_dim: int, default=10000
        Number of samples to be used in the validation set
    test_dim: int, default=10000
        Number of samples to be used in the test set
    ohe_values: tuple[int, int], default=(0, 32)
        Tuple containing the values to be used in the one-hot encoding of the labels

    Returns
    -------
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels): tuple[np.ndarray, ...]
        Tuple containing the training, validation and test data and labels
    """
    # Shuffle train set
    train_data, train_labels = train_set
    test_data, test_labels = test_set
    train_data, train_labels = shuffle_dataset(train_data, train_labels)

    if val_dim != 0:
        # Train-validation-test split
        train_data, val_data = train_data[:-val_dim], train_data[-val_dim:]
        train_labels, val_labels = train_labels[:-val_dim], train_labels[-val_dim:]
    else:
        # Train-test split
        val_data, val_labels = None, None

    test_data, test_labels = test_data[:test_dim], test_labels[:test_dim]

    return _preprocess_integer_dataset(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        ohe_values=ohe_values
    )


def _preprocess_integer_dataset(train_data, train_labels, val_data, val_labels, test_data, test_labels,
                                ohe_values: tuple[int, int] = (0, 32)) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """
    Scale the data in the correct range and apply one-hot encoding.

    Parameters
    ----------
    train_data: np.ndarray
        Training data
    train_labels: np.ndarray
        Training labels
    val_data: np.ndarray
        Validation data
    val_labels: np.ndarray
        Validation labels
    test_data: np.ndarray
        Test data
    test_labels: np.ndarray
        Test labels
    ohe_values: tuple[int, int], default=(0, 32)
        Tuple containing the values to be used in the one-hot encoding of the labels

    Returns
    -------
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels): tuple[np.ndarray, ...]
        Tuple containing the training, validation and test data and labels
    """
    # Train set (scaled to [-128, 127])
    # Convert to int16 so that no overflow occurs when normalizing the data
    train_data = np.subtract(train_data, 128, dtype=np.int16)
    if val_data is not None:
        val_data = np.subtract(val_data, 128, dtype=np.int16)
    test_data = np.subtract(test_data, 128, dtype=np.int16)

    # One-hot encoding of labels
    num_classes = len(np.unique(train_labels))
    train_labels = np.eye(num_classes, dtype=np.int8)[train_labels] * ohe_values[1]
    train_labels[train_labels == 0] = ohe_values[0]
    if val_data is not None:
        val_labels = np.eye(num_classes, dtype=np.int8)[val_labels] * ohe_values[1]
        val_labels[val_labels == 0] = ohe_values[0]
    test_labels = np.eye(num_classes, dtype=np.int8)[test_labels] * ohe_values[1]
    test_labels[test_labels == 0] = ohe_values[0]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


class Transform(ABC):
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def draw(self) -> bool:
        return np.random.binomial(1, self.probability) == 1

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class RandomCrop(Transform):
    def __init__(self, size: int, padding: int = 0, probability: float = 0.5):
        super().__init__(probability)
        self.size = size
        self.padding = padding

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.draw():
            # Apply padding
            pad_width = np.array([(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)])
            x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=to_cpu(x.min()))

            # Determine the original dimensions
            _, _, height, width = x.shape

            # Randomly choose the top-left pixel of the crop area
            top = np.random.randint(0, self.padding * 2 + 1)
            left = np.random.randint(0, self.padding * 2 + 1)

            # Perform the crop directly in the padded array
            x = x[:, :, top:top + self.size, left:left + self.size]

        return x

    def __str__(self) -> str:
        return f'RandomCrop(size={self.size}, padding={self.padding}, probability={self.probability})'


class RandomHorizontalFlip(Transform):
    def __init__(self, probability: float = 0.5):
        super().__init__(probability)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.flip(x, axis=3) if self.draw() else x

    def __str__(self) -> str:
        return f'RandomHorizontalFlip(probability={self.probability})'


class AugmentationPipeline:
    def __init__(self, transformations: list[Transform]):
        self.transformations = transformations

    def apply_augmentation(self, x_train):
        for transform in self.transformations:
            x_train = transform(x_train)
        return x_train

    def __call__(self, *args, **kwargs):
        return self.apply_augmentation(*args, **kwargs)

    def __str__(self) -> str:
        string = f'{self.__class__.__name__}([\n'
        for transform in self.transformations:
            string += f'    {transform}\n'
        string += '])'
        return string
