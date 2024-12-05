import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from lion_pytorch import Lion
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.classification import Accuracy


class Affine(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.A = nn.Parameter(torch.ones([1, 1, d_model]))
        self.b = nn.Parameter(torch.zeros([1, 1, d_model]))

    def forward(self, x):
        x = torch.einsum("d, b n d -> b n d", self.A, x) + self.b

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        self.forward_feature = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.forward_feature(x)

        return x


class ResMLPblock(nn.Module):
    def __init__(
        self,
        dim,
        num_patch,
        mlp_dim,
        dropout=0.0,
        init_values=1e-4,
    ):
        super().__init__()

        self.pre_affine = Affine(dim)
        self.token_mix = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Linear(num_patch, num_patch),
            Rearrange("b d n -> b n d"),
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)))

        self.post_affine = Affine(dim)
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
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


class ResMLP(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        img_size,
        patch_size,
        channels,
        depth,
        d_model,
        d_hidden,
        num_classes,
        init_values,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        assert (
            img_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patch = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(channels, d_model, patch_size, patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                ResMLPblock(d_model, self.num_patch, d_hidden, init_values)
            )

        self.affine = Affine(d_model)

        self.head = nn.Sequential(nn.Linear(d_model, num_classes))

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

        for block in self.blocks:
            x = block(x)
        x = self.affine(x)

        x = x.mean(dim=1)
        x = self.head(x)

        return x

    def forward(self, x):
        x = self.forward_compiled(x)

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
