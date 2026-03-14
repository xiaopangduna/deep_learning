import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from lovely_deep_learning.models.DAGNet import DAGNet


class ImageClassifierModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3, pretrained=True, model=None):
        """
        num_classes: 数据集类别数量
        lr: 学习率
        pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.model = DAGNet(model["structure"], model["weight"], pretrained)

    def forward(self, x):
        return self.model(x)

    def forward_batch(self, batch):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        class_id = net_out["class_id"]
        logits = self([img])
        loss = F.cross_entropy(logits[0], class_id)
        preds = logits[0].argmax(dim=1)
        acc = (preds == class_id).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward_batch(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward_batch(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        class_id = net_out["class_id"]
        
        logits = self([img])
        loss = F.cross_entropy(logits[0], class_id)
        class_id_pred = logits[0].argmax(dim=1)
        acc = (class_id_pred == class_id).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]

        return {"class_id_pred": class_id_pred, "class_id_conf": class_id_conf}

    def predict_step(self, batch, batch_idx):
        net_in, _ = batch
        img = net_in["img_tv_transformed"]
        logits = self([img])
        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]
        class_id_pred = logits[0].argmax(dim=1)
        return {"class_id_pred": class_id_pred, "class_id_conf": class_id_conf}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
