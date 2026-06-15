from .base import BaseModule


class ImageClassifierModule(BaseModule):
    def on_fit_start(self):
        self.model.train()
        return

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
