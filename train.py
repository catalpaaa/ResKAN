from dataclasses import asdict

import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

from model_config import ResKAN_config, ResMLP_config
from modules.ema import EMA, EMAModelCheckpoint
from ResKAN import ResKAN
from ResMLP import ResMLP


IMAGENET_ROOT = "./datasets/ImageNet 1k"


class dataset(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = datasets.ImageFolder(
            IMAGENET_ROOT + "/train",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        self.valid_set = datasets.ImageFolder(
            IMAGENET_ROOT + "/val",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


config = asdict(ResKAN_config())
data = dataset(batch_size=config["batchsize"], num_workers=8)
config.pop("batchsize")
model = ResKAN(**config)

# config = asdict(ResMLP_config())
# data = dataset(batch_size=config["batchsize"], num_workers=8)
# config.pop("batchsize")
# model = ResMLP(**config)

trainer = pl.Trainer(
    callbacks=[
        EMA(decay=0.9999),
        EMAModelCheckpoint(
            dirpath="models/",
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    logger=pl_loggers.WandbLogger(project="ResKAN", name="ResKAN"),
    precision="bf16-mixed",
    max_epochs=300,
)

trainer.fit(model, data)
