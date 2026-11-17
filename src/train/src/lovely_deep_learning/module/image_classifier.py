from .base import BaseModule


class ImageClassifierModule(BaseModule):
    def on_fit_start(self):
        self.model.train()
        return

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self.metrics.compute(stage)
        for name, value in metrics.items():
            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        self.metrics.reset(stage)

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
