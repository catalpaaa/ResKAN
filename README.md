# ResKAN

## About

RedMLP, but KAN.

## Installation

We provided a simple [setup.sh](setup.sh) to install the Conda environment. You need to satisfy the following prerequisite:

- Linux
- NVIDIA GPU
- CUDA 11+ supported GPU driver
- Miniforge

Then, simply run `source ./setup.sh` to get started.

## Pretrained Models

These models were trained on the [ImageNet-1k dataset](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) using a single RTX 4090 during our experiments.

| Name        | Model Dim. | Num. of Layers | Grid Size | Num. of Param. | Input Res. | Top-1 | Top-5 | Batch Size | Download | Training Log |
|-------------|------------|----------------|-----------|----------------|------------|-------|-------|------------|----------|--------------|
| ResKAN Tiny | 128        | 12             | 8         | 18.6           | 224²       | N/A   | N/A   | 256        | N/A      | N/A          |

## Training and inferencing

To set up the ImageNet-1k dataset, download both the training and validation sets.

We provide [train.py](train.py), which contains all the necessary code to train a ResKAN model and log the training progress. The logged parameters can be modified in [model.py](model.py).

The base model's hyperparameters are stored in [model_config.py](model_config.py), and you can adjust them as needed. When further training our model, note that all hyperparameters are saved directly in the model file. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint). The same applies to inferencing, as PyTorch Lightning automatically handles all parameters when loading our model.

Here's a sample code snippet to perform inferencing with ResKAN:

```python
import torch

from ResKAN import ResKAN

model = ResKAN.load_from_checkpoint("path_to.ckpt")
model.eval()

sample = torch.rand(3, 224, 224) # Channel, Width, Height
sample = sample.unsqueeze(0) # Batch, Channel, Width, Height
pred = model(sample) # Batch, Number of classes
```

Please note that the model's initial pass may be slower due to `torch.compile`.

## Credits

Our work builds upon the remarkable achievements of [ResMLP](https://arxiv.org/abs/2105.03404), and [KAN](https://arxiv.org/abs/2404.19756). [ResMLP.py](ResMLP.py) is our implementation of ResMLP using PyTorch Lightning.

[module/ema](modules/ema) is modified from [here](https://github.com/BioinfoMachineLearning/bio-diffusion/blob/main/src/utils/__init__.py).

[modules/lr_scheduler.py](modules/lr_scheduler.py) is taken from [here](https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/optimizers/lr_scheduler.py).
