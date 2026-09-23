from __future__ import annotations

from typing import Any

import torch

from .base import BaseModule, _class_path_cfg
from .yolo_optim import (
    apply_yolo_warmup,
    split_yolo_param_groups,
    yolo_linear_lf,
)


def standalone_eval_should_fuse(trainer: Any) -> bool:
    """仅独立 ``validate`` / ``test`` / ``predict`` 时 fuse。

    Lightning ``Trainer`` **没有** ``fitting`` 属性；旧代码 ``getattr(trainer, "fitting", False)``
    恒为 False，sanity check / fit 内 val 会把 Conv+BN 融掉，优化器还握着旧参数，主干不再更新。
    """
    if trainer is None:
        return False
    fn = getattr(getattr(trainer, "state", None), "fn", None)
    fn_val = getattr(fn, "value", fn)
    if fn_val in {"fit", "tuning"}:
        return False
    if getattr(trainer, "sanity_checking", False) or getattr(trainer, "training", False):
        return False
    return True


class ObjectDetectModule(BaseModule):
    """检测 LightningModule。默认不算训练集 mAP（COCO 全量会撑爆内存）；YAML 设 ``log_train_metrics: true`` 可打开。"""

    def __init__(
        self,
        log_train_metrics: bool = False,
        yolo_param_groups: bool = False,
        warmup_epochs: float = 3.0,
        warmup_bias_lr: float = 0.1,
        warmup_momentum: float = 0.8,
        lrf: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(log_train_metrics=log_train_metrics, **kwargs)
        self.yolo_param_groups = bool(yolo_param_groups)
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_bias_lr = float(warmup_bias_lr)
        self.warmup_momentum = float(warmup_momentum)
        self.lrf = float(lrf)
        self._yolo_lf = yolo_linear_lf(500, self.lrf)

    def configure_optimizers(self):
        """官方：pg0=bias 无 wd，pg1=conv 有 wd，pg2=BN 无 wd；线性 ``lrf`` 按 epoch。"""
        if not self.yolo_param_groups:
            return super().configure_optimizers()
        parsed = _class_path_cfg(self.optimizer_cfg)
        if parsed is None:
            raise TypeError("yolo_param_groups 需要 YAML optimizer 为 class_path 配置")
        args = dict(parsed.get("init_args") or {})
        lr = float(args.get("lr", 0.01))
        momentum = float(args.get("momentum", 0.937))
        nesterov = bool(args.get("nesterov", True))
        decay = float(args.get("weight_decay", 5.0e-4))
        g0, g1, g2 = split_yolo_param_groups(self)
        if not g2:
            raise RuntimeError("YOLO 参数分组未找到 bias，无法作为 pg0")
        optimizer = torch.optim.SGD(
            g2, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=0.0
        )
        optimizer.add_param_group({"params": g0, "weight_decay": decay})
        if g1:
            optimizer.add_param_group({"params": g1, "weight_decay": 0.0})
        epochs = 500
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            epochs = int(getattr(trainer, "max_epochs", epochs) or epochs)
        self._yolo_lf = yolo_linear_lf(epochs, self.lrf)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._yolo_lf)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not self.yolo_param_groups:
            return
        trainer = self.trainer
        if getattr(trainer, "sanity_checking", False):
            return
        nb = getattr(trainer, "num_training_batches", None)
        if nb is None or nb == float("inf"):
            return
        nb_i = int(nb)
        nw = (
            max(round(self.warmup_epochs * nb_i), 100)
            if self.warmup_epochs > 0
            else -1
        )
        ni = int(batch_idx) + nb_i * int(self.current_epoch)
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        apply_yolo_warmup(
            opt,
            ni=ni,
            nw=nw,
            epoch=int(self.current_epoch),
            lf=self._yolo_lf,
            warmup_bias_lr=self.warmup_bias_lr,
            warmup_momentum=self.warmup_momentum,
            momentum=float(opt.param_groups[0].get("momentum", 0.937)),
        )

    def _fuse_for_standalone_eval(self) -> None:
        """独立 ``validate`` / ``test`` 时 fuse；``fit`` 中的 val 不 fuse，以免拿掉 BN。"""
        if not standalone_eval_should_fuse(self.trainer):
            return
        fuse = getattr(self.model, "fuse", None)
        if callable(fuse):
            fuse()

    def on_validation_start(self) -> None:
        self._fuse_for_standalone_eval()

    def on_test_start(self) -> None:
        self._fuse_for_standalone_eval()

    def training_step(self, batch, batch_idx):
        """``train_loss`` 仍记未缩放值；返回给 Lightning 的 loss 乘 ``accumulate_grad_batches``。

        Lightning 自动优化会再把 loss 除以累积步数。官方 YOLO 是微批梯度 **求和** 再 step，
        乘回去后与 ``nbs=64``、``batch=16``、``accumulate=4`` 的步长一致。
        """
        loss, metric_preds, net_out, batch_size = self._shared_step(batch, "train")
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        acc = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
        return {
            "loss": loss * max(acc, 1),
            "metric_preds": metric_preds,
            "net_out": net_out,
        }

    def on_train_epoch_end(self):
        if not self.log_train_metrics:
            return
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
