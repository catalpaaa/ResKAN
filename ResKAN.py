import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from modules.fastkan import FastKAN
from lion_pytorch import Lion
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.classification import Accuracy


class Affine(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()

        self.A = nn.Parameter(torch.ones([d_in]))
        self.b = nn.Parameter(torch.zeros([1, 1, d_in]))

    def forward(self, x: torch.Tensor):
        x = torch.einsum("d, b n d -> b n d", self.A, x) + self.b

        return x


class ResKANblock(nn.Module):
    def __init__(
        self,
        num_patch: int,
        d_model: int,
        grid_size: int,
        expand: int,
        init_values: float,
    ):
        super().__init__()

        self.pre_affine = Affine(d_model)
        self.token_mix = nn.Sequential(
            Rearrange("b n d -> b d n"),
            FastKAN([num_patch, num_patch], num_grids=grid_size),
            Rearrange("b d n -> b n d", d=d_model),
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((d_model)))

        self.post_affine = Affine(d_model)
        self.ff = nn.Sequential(
            Rearrange("b n d -> b n d"),
            FastKAN([d_model, d_model * expand, d_model], num_grids=grid_size),
            Rearrange("b n d -> b n d", n=num_patch),
        )
        self.gamma_2 = nn.Parameter(init_values * torch.ones((d_model)))

    def forward(self, x: torch.Tensor):
        x = x + torch.einsum(
            "d, b n d -> b n d",
            self.gamma_1,
            self.token_mix(self.pre_affine(x)),
        )
        x = x + torch.einsum(
            "d, b n d -> b n d",
            self.gamma_2,
            self.ff(self.post_affine(x)),
        )

        return x


class ResKAN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        img_size: int,
        patch_size: int,
        channels: int,
        depth: int,
        d_model: int,
        grid_size: int,
        expand: int,
        num_classes: int,
        init_values: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        assert (
            img_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patch = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, d_model, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch, d_model))

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            block = ResKANblock(
                num_patch,
                d_model,
                grid_size,
                expand,
                init_values,
            )
            self.blocks.append(block)

        self.affine = Affine(d_model)

        self.head = FastKAN([d_model, num_classes], num_grids=grid_size)

        self.ce_loss = nn.CrossEntropyLoss()

        self.valid_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

    @torch.compile(mode="max-autotune")
    def forward_compiled(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)
        x = self.affine(x)

        x = x.mean(dim=1)
        x = self.head(x)

        return x

    def forward_uncompiled(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)
        x = self.affine(x)

        x = x.mean(dim=1)
        x = self.head(x)

        return x

    def forward(self, x: torch.Tensor, torch_compiled: bool = True):
        if torch_compiled:
            x = self.forward_compiled(x)
        else:
            x = self.forward_uncompiled(x)

        return x

    def training_step(self, batch):
        sample, target = batch
        preds = self(sample)

        loss = self.ce_loss(preds, target)

        self.log(
            "Training Loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        sample, target = batch
        preds = self(sample)

        loss = self.ce_loss(preds, target)

        self.log(
            "Validation Accuracy Top 1",
            self.valid_acc_top_1(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation Accuracy Top 5",
            self.valid_acc_top_5(preds, target),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Validation Loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = Lion(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            use_triton=True,
        )

        lr_scheduler_config = {
            "scheduler": CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "Learning Rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
