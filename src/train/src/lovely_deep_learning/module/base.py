from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import torch

from lovely_deep_learning.model.DAGNet import DAGNet


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

    def prune(self, ckpt_path: Optional[Union[str, Path]] = None) -> str:
        """剪枝导出；剪枝超参见 YAML ``pruner.init_args``，仅 ``ckpt_path`` 由 CLI 传入。"""
        if not isinstance(self.model, DAGNet):
            raise TypeError(
                f"{type(self).__name__}.prune 要求 `self.model` 为 DAGNet，当前为 {type(self.model).__name__!r}。"
            )
        if self.model.pruner is None:
            raise ValueError("pruner is None，请在 YAML 中配置 model.init_args.model.pruner。")
        if ckpt_path is None:
            raise ValueError("prune 需要 --ckpt_path。")
        return self.model.pruner.prune(self.model, ckpt_path=ckpt_path)
