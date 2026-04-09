"""
基于 ``DAGNet`` + ``configs/models/yolov8_n.yaml`` 的目标检测 Lightning 模块；
损失为 :class:`~lovely_deep_learning.loss.object_detect.DetectionLossYOLOv8`（
自研前向，与 Ultralytics ``v8DetectionLoss`` 公式对齐；便于对照源码与论文）。

数据侧使用 ``lovely_deep_learning.data_module.object_detect.ObjectDetectDataModule``。
实验 YAML 中请将 ``model.class_path`` 指向本模块的 ``ObjectDetectModule``。
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import instantiate_class
from typing import Any

from lovely_deep_learning.model.DAGNet import DAGNet

from ..loss.object_detect import DetectionLossYOLOv8
from ..metric.object_detect import ObjectDetectYOLOMetric


class ObjectDetectModule(pl.LightningModule):
    def __init__(
        self,
        model: Any = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        criterion: Any = None,
        metrics: Any = None,
    ):
        super().__init__()
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        if model is None:
            raise ValueError(
                "`model` config is required (DAGNet / yolov8_n.yaml 字段).")
        if self.optimizer_cfg is None:
            raise ValueError(
                "`optimizer` config is required in YAML (model.init_args.optimizer).")
        if self.lr_scheduler_cfg is None:
            raise ValueError(
                "`lr_scheduler` config is required in YAML (model.init_args.lr_scheduler).")
        if criterion is None:
            raise ValueError(
                "`criterion` config is required in YAML (model.init_args.criterion).")
        if metrics is None:
            raise ValueError(
                "`metrics` config is required in YAML (model.init_args.metrics).")

        self.model: DAGNet = DAGNet(**model)
        self.criterion: DetectionLossYOLOv8 = criterion
        self.metrics: ObjectDetectYOLOMetric = metrics

    def forward(self, x: torch.Tensor):
        return self.model([x])

    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"] for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self.model([imgs])
        loss = self.criterion(preds, net_out=net_out)

        with torch.inference_mode():
            self.metrics.update("train", preds, net_out)

        self.log("train_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        metrics = self.metrics.compute("train")
        self.log("train_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("train")

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"] for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self.model([imgs])
        loss = self.criterion(preds, net_out=net_out)

        with torch.inference_mode():
            self.metrics.update("val", preds, net_out)

        self.log("val_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute("val")
        self.log("val_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("val")

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"] for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self.model([imgs])
        loss = self.criterion(preds, net_out=net_out)

        with torch.inference_mode():
            detections = self.metrics.to_detections(preds)
            self.metrics.update("test", preds, net_out, detections=detections)

        self.log("test_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

        return {"detections": detections}

    def on_test_epoch_end(self):
        metrics = self.metrics.compute("test")
        self.log("test_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("test")

    def predict_step(self, batch, batch_idx):
        net_in, _net_out = batch
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        with torch.inference_mode():
            out = self.model([imgs])
            detections = self.metrics.to_detections(out)
        return {"detections": detections}

    def configure_optimizers(self):
        optimizer = instantiate_class(
            self.model.parameters(), self.optimizer_cfg)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
