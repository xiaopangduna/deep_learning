"""Shared export path: default output locations, PL ckpt → DAGNet, exporter.export.

第一版对外只暴露 ``ckpt_path``（可选）与 ``export_format``；由 :class:`~lovely_deep_learning.module.base.BaseModule` / CLI 传入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from lovely_deep_learning.model.DAGNet import DAGNet

_SUFFIX_BY_FORMAT: dict[str, str] = {
    "onnx": ".onnx",
    "pt": ".pt",
    "trt": ".engine",
    "tensorrt": ".engine",
    "engine": ".engine",
}


def normalize_export_format(fmt: str) -> str:
    f = str(fmt).lower()
    if f in ("trt", "tensorrt", "engine"):
        return "trt"
    return f


def compute_default_output_path(
    dagnet: DAGNet,
    *,
    ckpt_path: Optional[Union[str, Path]],
    export_format: str,
) -> Path:
    fmt = normalize_export_format(export_format)
    suffix = _SUFFIX_BY_FORMAT.get(fmt, f".{fmt}")
    if ckpt_path is not None:
        ckpt = Path(ckpt_path).resolve()
        return ckpt.parent / f"{ckpt.stem}{suffix}"
    model_name = getattr(dagnet, "model_name", "undefined")
    return Path.cwd() / f"{model_name}{suffix}"


def ensure_exporter_output_paths(
    dagnet: DAGNet,
    *,
    ckpt_path: Optional[Union[str, Path]] = None,
    export_format: str = "onnx",
) -> None:
    if dagnet.exporter is None:
        raise ValueError("DAGNet.exporter 为 None，请在 YAML 中配置 model.init_args.model.exporter。")
    fmt = normalize_export_format(export_format)
    default_out = compute_default_output_path(dagnet, ckpt_path=ckpt_path, export_format=fmt)
    s = str(default_out)
    if fmt == "pt":
        cfg = dict(dagnet.exporter.pt_cfg or {})
        cfg.setdefault("output_path", s)
        dagnet.exporter.pt_cfg = cfg
    elif fmt == "onnx":
        cfg = dict(dagnet.exporter.onnx_cfg or {})
        if cfg.get("path_save_onnx") is None:
            cfg["path_save_onnx"] = s
        cfg.setdefault("output_path", s)
        dagnet.exporter.onnx_cfg = cfg
    elif fmt == "trt":
        cfg = dict(dagnet.exporter.trt_cfg or {})
        cfg.setdefault("output_path", s)
        dagnet.exporter.trt_cfg = cfg


def load_pl_checkpoint_into_dagnet(dagnet: DAGNet, ckpt_path: Union[str, Path]) -> None:
    dagnet.load_weights(
        map_location="cpu",
        strict=False,
        stages=[{"format": "dense", "path": str(ckpt_path)}],
    )


def export_dagnet(
    dagnet: DAGNet,
    *,
    ckpt_path: Optional[Union[str, Path]] = None,
    export_format: str = "onnx",
) -> str:
    ensure_exporter_output_paths(dagnet, ckpt_path=ckpt_path, export_format=export_format)
    if ckpt_path is not None:
        load_pl_checkpoint_into_dagnet(dagnet, ckpt_path)
    fmt = normalize_export_format(export_format)
    if dagnet.exporter is None:
        raise ValueError("DAGNet.exporter 为 None，请在 YAML 中配置 model.init_args.model.exporter。")
    return dagnet.exporter.export(model=dagnet, export_format=fmt)
