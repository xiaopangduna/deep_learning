from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lovely_deep_learning.export.base import BaseExporter
from lovely_deep_learning.model.DAGNet import DAGNet


class _DAGNetExportWrapper(nn.Module):
    def __init__(self, model: DAGNet):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model([images])


class YOLOv8Exporter(BaseExporter):
    def __init__(
        self,
        onnx_cfg: dict[str, Any] | None = None,
        trt_cfg: dict[str, Any] | None = None,
        pt_cfg: dict[str, Any] | None = None,
    ):
        super().__init__(pt_cfg=pt_cfg)
        self.onnx_cfg = onnx_cfg or {}
        self.trt_cfg = trt_cfg or {}
        self.model: DAGNet | None = None

    def export(
        self,
        format: str = "onnx",
    ) -> str:
        if self.model is None:
            raise ValueError("exporter 尚未绑定模型，请先调用 bind(model)。")

        fmt = str(format).lower()
        if fmt == "trt":
            fmt = "engine"
        if fmt == "tensorrt":
            fmt = "engine"

        if fmt == "pt":
            return self.export_pt()
        if fmt == "onnx":
            return self.export_onnx()
        raise ValueError(f"当前仅支持 format=onnx/pt，收到: {fmt!r}")

    def export_onnx(self) -> str:
        if self.model is None:
            raise ValueError("exporter 尚未绑定模型，请先调用 bind(model)。")
        cfg = self.onnx_cfg
        final_input_shape = cfg.get("input_shape")
        if final_input_shape is None:
            raise ValueError("onnx_cfg.input_shape 为必填项。")
        if len(final_input_shape) != 4:
            raise ValueError(f"input_shape 须为 [N,C,H,W]，当前: {final_input_shape}")
        n, c, h, w = (int(x) for x in final_input_shape)
        dummy = torch.randn(n, c, h, w)

        output = cfg.get("output_path")
        if output is None:
            output = "model.onnx"
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        opset = int(cfg.get("opset", 17))
        self.model.eval()
        wrapper = _DAGNetExportWrapper(self.model)
        torch.onnx.export(
            wrapper,
            dummy,
            output,
            export_params=True,
            opset_version=opset,
            input_names=["images"],
        )
        return str(output)
