from __future__ import annotations

from .base import BaseModule


class ObjectDetectModule(BaseModule):
    def _fuse_for_standalone_eval(self) -> None:
        """独立 ``validate`` / ``test`` 时 fuse；``fit`` 中的 val 不 fuse，以免拿掉 BN。"""
        if getattr(self.trainer, "fitting", False):
            return
        fuse = getattr(self.model, "fuse", None)
        if callable(fuse):
            fuse()

    def on_validation_start(self) -> None:
        self._fuse_for_standalone_eval()

    def on_test_start(self) -> None:
        self._fuse_for_standalone_eval()

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
