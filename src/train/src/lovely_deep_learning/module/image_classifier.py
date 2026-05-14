from typing import Any

import torch
from lightning.pytorch.cli import instantiate_class

from .base import BaseModule


class ImageClassifierModule(BaseModule):
    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        img: torch.Tensor = net_in["img_tv_transformed"]
        batch_size = img.shape[0]
        preds = self(img)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            post_out = self.postprocess.run(preds)
            self.metrics.update("train", post_out, net_out)

        self.log("train_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

        out: dict[str, Any] = {"loss": loss}
        if batch_idx == 0:
            out["tb_sample"] = {
                "img": img,
                "net_out": net_out,
                "pred_ids": post_out["pred_ids"],
                "pred_conf": post_out["pred_conf"],
                "split": "train",
            }
        return out

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        batch_size = img.shape[0]
        preds = self(img)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            post_out = self.postprocess.run(preds)
            self.metrics.update("val", post_out, net_out)

        self.log("val_loss", loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

        out: dict[str, Any] = {"loss": loss}
        if batch_idx == 0:
            out["tb_sample"] = {
                "img": img,
                "net_out": net_out,
                "pred_ids": post_out["pred_ids"],
                "pred_conf": post_out["pred_conf"],
                "split": "val",
            }
        return out

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        targets = net_out["class_id"]

        preds = self(img)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        self.log("test_loss", loss,
                 batch_size=targets.shape[0], on_step=False, on_epoch=True, prog_bar=True)

        with torch.inference_mode():
            post_out = self.postprocess.run(preds)
            self.metrics.update("test", post_out, net_out)

        return {
            "class_id_pred": post_out["pred_ids"],
            "class_id_conf": post_out["pred_conf"],
        }

    def predict_step(self, batch, batch_idx):
        net_in, _ = batch
        img = net_in["img_tv_transformed"]
        with torch.inference_mode():
            preds = self(img)
            post_out = self.postprocess.run(preds)
        return {
            "class_id_pred": post_out["pred_ids"],
            "class_id_conf": post_out["pred_conf"],
        }

    def configure_optimizers(self):
        optimizer = instantiate_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.optimizer_cfg,
        )
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self):
        pass

    def on_train_epoch_end(self):
        metrics = self.metrics.compute("train")
        self.log(
            "train_acc",
            metrics["acc"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.metrics.reset("train")

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute("val")
        self.log(
            "val_acc",
            metrics["acc"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.metrics.reset("val")

    def on_test_epoch_end(self):
        metrics = self.metrics.compute("test")
        self.log(
            "test_acc",
            metrics["acc"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.metrics.reset("test")
