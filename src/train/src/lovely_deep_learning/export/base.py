from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class BaseExporter:
    """通用导出基类：提供 PT 导出与统一入口。"""

    def __init__(
        self,
        pt_cfg: dict[str, Any] | None = None,
        onnx_cfg: dict[str, Any] | None = None,
        trt_cfg: dict[str, Any] | None = None,
        **_: Any,
    ):
        self.pt_cfg = pt_cfg or {}
        # Keep for config compatibility; BaseExporter currently only implements PT export.
        self.onnx_cfg = onnx_cfg or {}
        self.trt_cfg = trt_cfg or {}
        self.model: Any | None = None

    def bind(self, model: Any) -> None:
        self.model = model

    def export(self, format: str = "pt") -> str:
        if self.model is None:
            raise ValueError("exporter 尚未绑定模型，请先调用 bind(model)。")
        fmt = str(format).lower()
        if fmt == "pt":
            return self.export_pt()
        raise ValueError(f"当前不支持 format={fmt!r}")

    def export_pt(self) -> str:
        if self.model is None:
            raise ValueError("exporter 尚未绑定模型，请先调用 bind(model)。")
        cfg = self.pt_cfg
        output = cfg.get("output_path")
        if output is None:
            output = "model.pt"
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        payload = {
            "state_dict": self.model.state_dict(),
            "model_name": getattr(self.model, "model_name", "undefined"),
            "structure": getattr(self.model, "structure_config", None),
        }
        torch.save(payload, output)
        return str(output)
