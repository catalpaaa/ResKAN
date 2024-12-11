from dataclasses import dataclass


@dataclass
class ResMLP_config:
    learning_rate: float = 5e-4
    weight_decay: float = 0.5
    batchsize: int = 256
    img_size: int = 224
    patch_size: int = 16
    channels: int = 3
    depth: int = 12
    d_model: int = 384
    d_hidden: int = 1536
    num_classes: int = 100
    dropout: float = 0.1
    init_values: float = 0.1


@dataclass
class ResKAN_config:
    learning_rate: float = 5e-4
    weight_decay: float = 0.5
    batchsize: int = 256
    img_size: int = 224
    patch_size: int = 16
    channels: int = 3
    depth: int = 12
    d_model: int = 128
    grid_size: int = 8
    expand: int = 4
    num_classes: int = 100
    init_values: float = 0.1
