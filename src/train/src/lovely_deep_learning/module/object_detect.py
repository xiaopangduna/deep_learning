from __future__ import annotations

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import instantiate_class
from typing import Any

from lovely_deep_learning.model.DAGNet import DAGNet

from ..loss.object_detect import DetectionLossYOLOv8
from ..metric.object_detect import ObjectDetectMetric
from ..postprocess.yolov8 import YOLOv8PostProcessor


class ObjectDetectModule(pl.LightningModule):
    def __init__(
        self,
        model: Any = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        criterion: Any = None,
        postprocess: Any = None,
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
        if postprocess is None:
            raise ValueError(
                "`postprocess` config is required in YAML (model.init_args.postprocess).")
        if metrics is None:
            raise ValueError(
                "`metrics` config is required in YAML (model.init_args.metrics).")

        self.model: DAGNet = DAGNet(**model)
        self.criterion: DetectionLossYOLOv8 = criterion
        self.postprocess: YOLOv8PostProcessor = postprocess
        self.metrics: ObjectDetectMetric = metrics

    def forward(self, x: torch.Tensor):
        return self.model([x])

    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"]
                           for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self(imgs)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            map_preds = self.postprocess.run(preds)
            self.metrics.update("train", map_preds, net_out)

        self.log("train_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "map_preds": map_preds, "net_out": net_out}

    def on_train_epoch_end(self):
        metrics = self.metrics.compute("train")
        self.log("train_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("train")

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"]
                           for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self(imgs)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            map_preds = self.postprocess.run(preds)
            self.metrics.update("val", map_preds, net_out)

        self.log("val_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"map_preds": map_preds, "net_out": net_out}

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute("val")
        self.log("val_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("val")

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"]
                           for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self(imgs)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            map_preds = self.postprocess.run(preds)
            self.metrics.update("test", map_preds, net_out)

        self.log("test_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"map_preds": map_preds, "net_out": net_out}

    def on_test_epoch_end(self):
        metrics = self.metrics.compute("test")
        self.log("test_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("test")

    def predict_step(self, batch, batch_idx):
        net_in, _net_out = batch
        imgs = torch.stack([item["img"] for item in net_in], dim=0)
        with torch.inference_mode():
            out = self(imgs)
            map_preds = self.postprocess.run(out)
        return {"map_preds": map_preds}

    def configure_optimizers(self):
        optimizer = instantiate_class(
            self.model.parameters(), self.optimizer_cfg)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
