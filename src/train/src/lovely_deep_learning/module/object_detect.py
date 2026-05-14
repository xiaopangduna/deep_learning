from __future__ import annotations

import torch
from lightning.pytorch.cli import instantiate_class

from .base import BaseModule


class ObjectDetectModule(BaseModule):
    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = torch.stack([item["img_tv_transformed"]
                           for item in net_in], dim=0)
        batch_size = imgs.shape[0]

        preds = self(imgs)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            metric_preds = self.postprocess.run(preds)
            self.metrics.update("train", metric_preds, net_out)

        self.log("train_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "metric_preds": metric_preds, "net_out": net_out}

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
            metric_preds = self.postprocess.run(preds)
            self.metrics.update("val", metric_preds, net_out)

        self.log("val_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"metric_preds": metric_preds, "net_out": net_out}

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
            metric_preds = self.postprocess.run(preds)
            self.metrics.update("test", metric_preds, net_out)

        self.log("test_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {"metric_preds": metric_preds, "net_out": net_out}

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
            metric_preds = self.postprocess.run(out)
        return {"metric_preds": metric_preds}

    def configure_optimizers(self):
        optimizer = instantiate_class(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            self.optimizer_cfg,
        )
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
