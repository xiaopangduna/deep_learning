from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.cli import instantiate_class
import torch


from lovely_deep_learning.model.DAGNet import DAGNet
from lovely_deep_learning.dataset.base import BaseDataset


class BaseModule(pl.LightningModule):
    """项目内 LightningModule 基类。

    **YAML 共性**（与检测 / 分类实验对齐）：由 Lightning CLI 注入 ``model``、
    ``optimizer``、``lr_scheduler``、``criterion``、``postprocess``、``metrics``，
    在本类 ``__init__`` 中校验并挂载为 ``self.*``。

    导出（第一版）：仅 ``ckpt_path``（可选）与 ``export_format``；其余依赖 YAML 中的 ``exporter``。
    """

    def __init__(
        self,
        model: Dict[str, Any],
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        criterion: Any = None,
        postprocess: Any = None,
        metrics: Any = None,
    ) -> None:
        super().__init__()
        if model is None:
            raise ValueError(
                "`model` config is required in YAML (model.init_args.model / DAGNet)."
            )
        if optimizer is None:
            raise ValueError(
                "`optimizer` config is required in YAML (model.init_args.optimizer)."
            )
        if lr_scheduler is None:
            raise ValueError(
                "`lr_scheduler` config is required in YAML (model.init_args.lr_scheduler)."
            )
        if criterion is None:
            raise ValueError(
                "`criterion` config is required in YAML (model.init_args.criterion)."
            )
        if postprocess is None:
            raise ValueError(
                "`postprocess` config is required in YAML (model.init_args.postprocess)."
            )
        if metrics is None:
            raise ValueError(
                "`metrics` config is required in YAML (model.init_args.metrics)."
            )

        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.model: DAGNet = DAGNet(**model)
        self.criterion = criterion
        self.postprocess = postprocess
        self.metrics = metrics

    def forward(self, x: torch.Tensor) -> Any:
        return self.model([x])

    def _shared_step(
        self, batch: Any, stage: str
    ) -> tuple[torch.Tensor, Any, Any, int]:
        net_in, net_out = batch
        imgs = BaseDataset.stack_batch_images(net_in)
        batch_size = imgs.shape[0]
        preds = self(imgs)
        loss = self.criterion(preds, net_out=net_out, net_in=net_in)

        with torch.inference_mode():
            metric_preds = self.postprocess.run(preds)
            self.metrics.update(stage, metric_preds, net_out)

        return loss, metric_preds, net_out, batch_size

    def training_step(self, batch, batch_idx):
        loss, metric_preds, net_out, batch_size = self._shared_step(
            batch, "train")
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "metric_preds": metric_preds, "net_out": net_out}

    def validation_step(self, batch, batch_idx):
        loss, metric_preds, net_out, batch_size = self._shared_step(
            batch, "val")
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"metric_preds": metric_preds, "net_out": net_out}

    def test_step(self, batch, batch_idx):
        loss, metric_preds, net_out, batch_size = self._shared_step(
            batch, "test")
        self.log(
            "test_loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"metric_preds": metric_preds, "net_out": net_out}

    def predict_step(self, batch, batch_idx):
        net_in, _net_out = batch
        imgs = BaseDataset.stack_batch_images(net_in)
        with torch.inference_mode():
            preds = self(imgs)
            metric_preds = self.postprocess.run(preds)
        return {"metric_preds": metric_preds}

    def export(
        self,
        ckpt_path: Optional[Union[str, Path]] = None,
        export_format: str = "onnx",
    ) -> str:
        """返回导出产物路径。``ckpt_path`` 仅加载 ``model.*`` 权重进 DAGNet，语义见 ``LovelyTrainer`` 类文档。"""
        model = getattr(self, "model", None)
        if not isinstance(model, DAGNet):
            raise TypeError(
                f"{type(self).__name__}.export（第一版）要求 `self.model` 为 DAGNet，当前为 {type(model).__name__!r}。"
                "请覆写 export() 或调整模块结构。"
            )
        from lovely_deep_learning.export.pipeline import export_dagnet

        return export_dagnet(model, ckpt_path=ckpt_path, export_format=export_format)

    def configure_optimizers(self):
        optimizer = instantiate_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.optimizer_cfg,
        )
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def prune(self, ckpt_path: Optional[Union[str, Path]] = None) -> str:
        """剪枝导出；剪枝超参见 YAML ``pruner.init_args``，仅 ``ckpt_path`` 由 CLI 传入。"""
        if not isinstance(self.model, DAGNet):
            raise TypeError(
                f"{type(self).__name__}.prune 要求 `self.model` 为 DAGNet，当前为 {type(self.model).__name__!r}。"
            )
        if self.model.pruner is None:
            raise ValueError(
                "pruner is None，请在 YAML 中配置 model.init_args.model.pruner。")
        if ckpt_path is None:
            raise ValueError("prune 需要 --ckpt_path。")
        return self.model.pruner.prune(self.model, ckpt_path=ckpt_path)
