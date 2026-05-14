from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class DAGNetExporterWrapper(nn.Module):
    """Wraps any module whose forward accepts a list of tensors (e.g. DAGNet)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model([images])

class BaseExporter:
    """通用导出基类：提供 PT 导出与统一入口。"""

    def __init__(
        self,
        pt_cfg: dict[str, Any] | None = None,
        onnx_cfg: dict[str, Any] | None = None,
        trt_cfg: dict[str, Any] | None = None,
    ):
        self.pt_cfg = pt_cfg or {}
        self.onnx_cfg = onnx_cfg or {}
        self.trt_cfg = trt_cfg or {}

    def export(self, model: Any, export_format: str = "pt") -> str:
        if model is None:
            raise ValueError("exporter 尚未绑定模型，传入模型。")
        fmt = str(export_format).lower()
        if fmt == "pt":
            return self.export_pt(model)
        elif fmt == "onnx":
            return self.export_onnx(model)
        elif fmt == "trt":
            return self.export_trt(model)

        raise ValueError(f"当前不支持 export_format={fmt!r}")

    def export_pt(self, model: Any) -> str:

        cfg = self.pt_cfg
        output = cfg.get("output_path")
        if output is None:
            output = "model.pt"
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        payload = {
            "state_dict": model.state_dict(),
            "model_name": getattr(model, "model_name", "undefined"),
            "structure": getattr(model, "structure_config", None),
        }
        torch.save(payload, output)
        return str(output)

    def export_onnx(self, model: Any) -> str:

        cfg = self.onnx_cfg
        input_shape = cfg.get("input_shape")
        if len(input_shape) != 4:
            raise ValueError(
                f"input_shape 须为 [N,C,H,W]，当前: {input_shape}")
        n, c, h, w = (int(x) for x in input_shape)
        dummy = torch.randn(n, c, h, w)

        output = cfg.get("path_save_onnx") or cfg.get("output_path")
        if output is None:
            raise ValueError("onnx_cfg 须设置 path_save_onnx 或 output_path。")
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.eval()
        wrapper = DAGNetExporterWrapper(model)
        torch.onnx.export(
            wrapper,
            dummy,
            output,
            **cfg.get("cfg_for_torch_onnx_export", {}),
        )
        return str(output)

    def export_trt(self) -> str:

        pass
