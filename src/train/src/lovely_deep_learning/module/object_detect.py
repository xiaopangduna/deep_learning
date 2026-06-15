from __future__ import annotations

from .base import BaseModule


class ObjectDetectModule(BaseModule):
    def on_train_epoch_end(self):
        metrics = self.metrics.compute("train")
        self.log("train_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("train")

    def on_validation_epoch_end(self):
        metrics = self.metrics.compute("val")
        self.log("val_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("val")

    def on_test_epoch_end(self):
        metrics = self.metrics.compute("test")
        self.log("test_map", metrics["map"],
                 prog_bar=True, on_step=False, on_epoch=True)
        self.metrics.reset("test")
