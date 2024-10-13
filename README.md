# NITRO-D: Native Integer-only Training of Deep Convolutional Neural Networks

This repository is the official implementation of [NITRO-D: Native Integer-only Training of Deep Convolutional Neural Networks](https://arxiv.org/abs/2407.11698).

NITRO-D is a novel training framework for arbitrarily deep **integer-only** Convolutional Neural Networks (CNNs) that operates entirely in the integer-only domain for **both training and inference**. NITRO-D utilizes a unique learning algorithm derived from Local Error Signals (LES) and it represents the first work in the literature to enable the training of integer-only deep CNNs **without the need to introduce a quantization scheme**.

![NITRO-D architecture](/figures/nitro_d_architecture.svg)

NITRO-D introduces both a **novel architecture** and a **novel integer-only learning algorithm** designed to train this architecture exploiting IntegerSGD, an optimizer designed specifically to operate in an integer-only context. Experimental evaluations demonstrate its effectiveness across several state-of-the-art image recognition datasets, highlighting considerable **performance improvements** from 2.47% to 5.96% over the state-of-the-art.

BibTeX entry:

```bibtex
@misc{pirillo2024nitrodnativeintegeronlytraining,
    title={NITRO-D: Native Integer-only Training of Deep Convolutional Neural Networks}, 
    author={Alberto Pirillo and Luca Colombo and Manuel Roveri},
    year={2024},
    eprint={2407.11698},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2407.11698}, 
}
```

## Requirements

NITRO-D was developed and tested using **Python 3.11** and **CUDA 12.1**.

Install the required packages using the following commands:

```shell
python -m pip install -U setuptools pip
pip install -r requirements.txt
```

*(optional)* Install *cuTensor* and *cuDNN* for optimal performance:

```shell
python -m cupyx.tools.install_library --cuda 12.x --library cutensor
python -m cupyx.tools.install_library --cuda 12.x --library cudnn
```

*(optional)* Enable the installed accelerators by setting environment variables. On Linux:

```shell
echo 'export CUPY_ACCELERATORS=cutensor,cub' >> ~/.bashrc
```

## Training

We provide a separate notebook to reproduce the results for all the configurations considered in the paper.
These notebooks are called `train.ipynb` and are located in the [`results`](/results/) directory.
Each notebook:

- Imports the required libraries
- Defines the experimental setup
- Loads and pre-processes the dataset
- Instantiates and trains the model `N_EXPERIMENTS` times with different initializations
- Saves the results of each run in a CSV file
- Saves the model of the first run in the `model.pkl` file
- Computes the average and standard deviation of the train and test accuracy

We also provide two notebooks in the [`examples`](/examples/) directory, [`nitro_cnn.ipynb`](/examples/nitro_cnn.ipynb) and [`nitro_mlp.ipynb`](/examples/nitro_mlp.ipynb), that detail the training and evaluation of NITRO-D models.

## Evaluation and pre-trained models

The models are automatically evaluated inside of the `train.ipnyb` notebooks. We also provide an additional [`eval.ipynb`](/examples/eval.ipynb) notebook in the [`examples`](/examples/) directory, which shows how to load a pre-trained model and evaluate it on a dataset.

## Results

NITRO-D models were trained and evaluated on three different datasets: MNIST, FashionMNIST, and CIFAR-10. The results are summarized in the table below, which reports the average test accuracy and the standard deviation over 10 runs.

| Model name                   | Dataset       | NITRO-D          | [PocketNN](https://export.arxiv.org/abs/2201.02863v2)| [FP LES](https://arxiv.org/abs/1901.06656) | FP BP   |
| :--------------------------- | :------------ | :--------------: | :------: | :-----: | :-----: |
| MLP [784-100-50-10]          | MNIST         | $97.36 \pm 0.23$ | $96.98$  | -       | $98.00$ |
| MLP [784-200-100-50-10]      | FashionMNIST  | $88.66 \pm 0.46$ | $87.70$  | -       | $89.79$ |
| MLP [1024-3000-3000-3000-10] | CIFAR-10      | $61.03 \pm 0.60$ | -        | $67.70$ | $66.40$ |
| VGG8B                        | MNIST         | $99.45 \pm 0.05$ | -        | $99.60$ | $99.74$ |
| VGG8B                        | FashionMNIST  | $93.66 \pm 0.40$ | -        | $94.34$ | $95.47$ |
| VGG8B                        | CIFAR-10      | $87.96 \pm 0.39$ | -        | $91.60$ | $94.01$ |
| VGG11B                       | CIFAR-10      | $87.39 \pm 0.64$ | -        | $91.61$ | $94.44$ |

## Contributing

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">NITRO-D</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
