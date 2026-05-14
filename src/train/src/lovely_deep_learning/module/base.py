from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl

from lovely_deep_learning.model.DAGNet import DAGNet


class BaseModule(pl.LightningModule):
    """项目内 LightningModule 基类。

    导出（第一版）：仅 ``ckpt_path``（可选）与 ``export_format``；其余依赖 YAML 中的 ``exporter``。
    """

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
