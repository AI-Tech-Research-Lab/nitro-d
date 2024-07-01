from enum import auto, Enum


class NonLinearity(Enum):
    POCKET_RELU = auto()
    POCKET_TANH = auto()
    POCKET_LEAKY_RELU = auto()
    NITRO_LEAKY_RELU = auto()
    POCKET_BIPOLAR_LEAKY_RELU = auto()
    POCKET_BIPOLAR_RELU = auto()


class Initialization(Enum):
    ZEROS = auto()
    ONES = auto()
    UNIFORM_STD = auto()
    UNIFORM_FIXED = auto()
    UNIFORM_KAIMING_LEAKY_RELU = auto()
    UNIFORM_KAIMING_RELU = auto()
    UNIFORM_XAVIER = auto()


class OptimizerName(Enum):
    INTEGER_SGD = auto()


class Dataset(Enum):
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()
    CIFAR10_GRAY = auto()
    CIFAR100 = auto()
    WHITE_NOISE = auto()
