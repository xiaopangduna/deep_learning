import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ImageClassifierModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, pretrained=True):
        """
        num_classes: 数据集类别数量
        lr: 学习率
        pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()
        self.save_hyperparameters()  # 保存超参到 checkpoint

        self.model = resnet50(pretrained=pretrained)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # 可选：学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]