from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _maybe_instantiate_module(obj: Any) -> nn.Module | None:
    """YAML ``class_path`` dict → ``nn.Module``；已是模块或 ``None`` 则原样返回。"""
    if obj is None or isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict) and "class_path" in obj:
        from lightning.pytorch.cli import instantiate_class

        inst = instantiate_class((), obj)
        if not isinstance(inst, nn.Module):
            raise TypeError(
                f"export_head 须为 nn.Module，实例化得到 {type(inst).__name__}"
            )
        return inst
    raise TypeError(
        f"export_head 须为 nn.Module、None 或带 class_path 的 dict，收到 {type(obj).__name__}"
    )


class ExportGraph(nn.Module):
    """``images`` 张量 → DAGNet（list 输入）→ 可选 ``export_head``。"""

    def __init__(self, model: nn.Module, head: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, images: torch.Tensor):
        dag_out = self.model([images])
        if self.head is None:
            return dag_out
        return self.head(dag_out[0])


class DAGNetExporterWrapper(ExportGraph):
    """兼容旧名：仅包 DAGNet 的 list 输入，不接 ``export_head``。"""

    def __init__(self, model: nn.Module):
        super().__init__(model, head=None)


def simplify_onnx(path: str | Path) -> None:
    """用 onnxslim 简化已写出的 ONNX（覆盖原文件）。``simplify=true`` 时失败即抛错。"""
    path = Path(path)
    try:
        import onnx
        import onnxslim
    except ImportError as e:
        raise ImportError(
            "onnx_cfg.simplify=true 需要 onnxslim，请执行: poetry add onnxslim==0.1.65"
        ) from e
    model = onnx.load(str(path))
    slimmed = onnxslim.slim(model)
    if slimmed is None:
        raise RuntimeError(f"onnxslim.slim 返回 None，未写出简化模型: {path}")
    onnx.save(slimmed, str(path))


class BaseExporter:
    """通用导出：PT / ONNX。``export_head`` 有则接到 DAGNet 输出之后，无则只导网络本身。"""

    def __init__(
        self,
        export_head: nn.Module | None = None,
        pt_cfg: dict[str, Any] | None = None,
        onnx_cfg: dict[str, Any] | None = None,
        trt_cfg: dict[str, Any] | None = None,
    ):
        self.export_head = _maybe_instantiate_module(export_head)
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
        if self.export_head is not None:
            self.export_head.eval()
        graph = ExportGraph(model, head=self.export_head)
        torch.onnx.export(
            graph,
            dummy,
            output,
            **cfg.get("cfg_for_torch_onnx_export", {}),
        )
        if cfg.get("simplify"):
            simplify_onnx(output)
        return str(output)

    def export_trt(self) -> str:

        pass
