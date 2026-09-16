from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.cli import instantiate_class
import torch
from torch.optim.lr_scheduler import LRScheduler

from lovely_deep_learning.model.DAGNet import DAGNet
from lovely_deep_learning.dataset.base import BaseDataset


def _class_path_cfg(spec: Any) -> dict[str, Any] | None:
    """Lightning YAML / jsonargparse ``Namespace`` → ``{class_path, init_args}``。"""
    if isinstance(spec, dict) and "class_path" in spec:
        init_args = spec.get("init_args") or {}
        if hasattr(init_args, "as_dict"):
            init_args = init_args.as_dict()
        return {"class_path": spec["class_path"], "init_args": dict(init_args)}
    class_path = getattr(spec, "class_path", None)
    if not class_path:
        return None
    init_args = getattr(spec, "init_args", None) or {}
    if hasattr(init_args, "as_dict"):
        init_args = init_args.as_dict()
    elif not isinstance(init_args, dict):
        init_args = vars(init_args)
    return {"class_path": str(class_path), "init_args": dict(init_args)}


def _bind_scheduler_to_optimizer(optimizer: Any, spec: Any) -> Any:
    """把 YAML 里的调度器配置或 jsonargparse 已实例化的子调度器绑到 ``optimizer``。"""
    cfg = _class_path_cfg(spec)
    if cfg is not None:
        return _instantiate_lr_scheduler(optimizer, cfg)
    if isinstance(spec, LRScheduler):
        kwargs: dict[str, Any] = {}
        for key in (
            "start_factor",
            "end_factor",
            "total_iters",
            "gamma",
            "last_epoch",
            "milestones",
            "T_max",
            "eta_min",
        ):
            if hasattr(spec, key):
                kwargs[key] = getattr(spec, key)
        return type(spec)(optimizer, **kwargs)
    raise TypeError(
        f"无法为优化器绑定学习率调度器，收到 {type(spec)!r}"
    )


def _instantiate_lr_scheduler(optimizer: Any, cfg: Any) -> Any:
    """``instantiate_class(optimizer, cfg)``；``SequentialLR`` / ``ChainedScheduler`` 的子调度器也要绑同一优化器。"""
    parsed = _class_path_cfg(cfg)
    if parsed is None:
        return cfg
    class_path = str(parsed["class_path"])
    init_args = dict(parsed.get("init_args") or {})
    init_args.pop("optimizer", None)
    if class_path.rsplit(".", 1)[-1] in {"SequentialLR", "ChainedScheduler"}:
        init_args["schedulers"] = [
            _bind_scheduler_to_optimizer(optimizer, sub)
            for sub in init_args.get("schedulers", [])
        ]
    parsed = {"class_path": class_path, "init_args": init_args}
    return instantiate_class(optimizer, parsed)


class BaseModule(pl.LightningModule):
    """分类 / 检测共用的 LightningModule 基类。

    **YAML 注入**（Lightning CLI → ``model.init_args``）：``model``（DAGNet 结构）、
    ``optimizer``、``lr_scheduler``、``criterion``、``postprocess``、``metrics``；
    在 ``__init__`` 中校验非空并挂到 ``self.*``。

    **Batch 约定**（与 ``BaseDataset.collate_net_in_tuple`` 对齐）::

        net_in:  tuple[dict]   # 每样本含 img_path / img_tv_transformed / img 等
        net_out: dict | tuple  # 分类为 batched dict；检测为 tuple[dict]（变长 GT）

    **Step 返回值**（供可视化 / Writer 等 Callback 消费）：

    - ``training_step`` → ``{"loss", "metric_preds", "net_out"}``
    - ``validation_step`` / ``test_step`` → ``{"metric_preds", "net_out"}``
    - ``predict_step`` → ``{"metric_preds"}``

    **子类职责**：本类已实现完整 train/val/test/predict 循环；子类通常只需覆写
    ``on_*_epoch_end`` 记录任务指标（如 acc / mAP），必要时覆写 ``on_fit_start`` 等。

    **导出 / 剪枝**：``export()``、``prune()`` 委托 DAGNet 与 YAML 中的 exporter / pruner。
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
        """``LightningModule`` 入口：单张量 batch 包装为 DAGNet 所需的 list 输入。"""
        return self.model([x])

    def _shared_step(
        self, batch: Any, stage: str
    ) -> tuple[torch.Tensor, Any, Any, int]:
        """train / val / test 共用：前向 → loss → postprocess → 更新 metrics。

        Returns:
            ``(loss, metric_preds, net_out, batch_size)``。
            ``metric_preds`` 形态由 YAML 中的 ``postprocess`` 决定（分类为 batched dict，
            检测为每图 list[dict]）。
        """
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
        """记录 ``train_loss``，返回 loss 与后处理结果供 Callback 使用。"""
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
        """记录 ``val_loss``；loss 不放入返回值（与检测侧 Callback 约定一致）。"""
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
        """记录 ``test_loss`` 并更新 test 阶段 metrics。"""
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
        """推理阶段不算 loss / metrics，仅 postprocess 后返回 ``metric_preds``。"""
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
        """优化 ``self.parameters()`` 下全部可训练子模块；``monitor`` 供 LR scheduler 使用。"""
        optimizer = instantiate_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            self.optimizer_cfg,
        )
        scheduler = _instantiate_lr_scheduler(optimizer, self.lr_scheduler_cfg)
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
