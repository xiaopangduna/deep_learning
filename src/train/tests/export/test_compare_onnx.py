from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from lovely_deep_learning.export.base import BaseExporter
from lovely_deep_learning.export.compare import (
    ImageInput,
    OnnxInfo,
    TensorSpec,
    compare_arrays,
    compare_onnx,
    io_errors,
    require_image_input,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_onnx_on_image.py"


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return (self.conv(x[0]),)


class _TwoOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        y = self.conv(x[0])
        return y, y + 1


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_onnx_on_image", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _export(path, model, *, output_names: list[str] | None = None, export_head=None) -> None:
    cfg_export = {
        "export_params": True,
        "opset_version": 13,
        "input_names": ["images"],
    }
    if output_names is not None:
        cfg_export["output_names"] = output_names
    BaseExporter(
        export_head=export_head,
        onnx_cfg={
            "input_shape": [1, 3, 64, 64],
            "path_save_onnx": str(path),
            "cfg_for_torch_onnx_export": cfg_export,
        },
    ).export_onnx(model)


def _info(path: str, *, channels: int = 3, height: int = 64, outputs: tuple[TensorSpec, ...] = ()) -> OnnxInfo:
    return OnnxInfo(
        path=path,
        nodes=1,
        opset=(13,),
        input=TensorSpec("images", (1, channels, height, 64), "float32"),
        outputs=outputs,
    )


def test_require_image_input_rejects_dynamic_spatial():
    info = OnnxInfo(
        path="a.onnx",
        nodes=1,
        opset=(13,),
        input=TensorSpec("images", (1, 3, None, 64), "float32"),
        outputs=(),
    )
    with pytest.raises(ValueError, match="高宽"):
        require_image_input(info)


def test_io_errors_reports_shape_and_output_count():
    left = _info("a.onnx", outputs=(TensorSpec("output0", (1, 4), "float32"),))
    right = _info(
        "b.onnx",
        height=32,
        outputs=(
            TensorSpec("a", (1, 4), "float32"),
            TensorSpec("b", (1, 4), "float32"),
        ),
    )
    errors = io_errors(
        left,
        right,
        ImageInput("images", (1, 3, 64, 64), "float32"),
        ImageInput("images", (1, 3, 32, 64), "float32"),
    )
    assert any("输入形状" in item for item in errors)
    assert any("输出个数" in item for item in errors)


def test_compare_arrays_allclose():
    ref = np.zeros((1, 4, 8), dtype=np.float32)
    other = ref + 1e-6
    assert compare_arrays(ref, other).ok
    assert not compare_arrays(ref, other + 1.0).ok


def test_compare_onnx_same_graph_passes(tmp_path):
    left = tmp_path / "a.onnx"
    right = tmp_path / "b.onnx"
    _export(left, _TinyNet(), output_names=["output0"])
    right.write_bytes(left.read_bytes())
    result = compare_onnx(left, right)
    assert result.ok
    assert result.image is None
    assert result.feed is not None
    assert result.feed.shape == (1, 3, 64, 64)
    assert result.outputs[0].max_abs == 0.0


def test_compare_onnx_output_count(tmp_path):
    left = tmp_path / "a.onnx"
    right = tmp_path / "b.onnx"
    _export(left, _TinyNet(), output_names=["output0"])
    _export(right, _TwoOut(), output_names=["output0", "output1"])
    result = compare_onnx(left, right)
    assert not result.ok
    assert result.feed is None
    assert any("输出个数" in item for item in result.errors)


def test_compare_onnx_missing_file(tmp_path):
    other = tmp_path / "b.onnx"
    _export(other, _TinyNet(), output_names=["output0"])
    with pytest.raises(FileNotFoundError, match="找不到参考模型"):
        compare_onnx(tmp_path / "no.onnx", other)


def test_compare_onnx_image_resize(tmp_path):
    left = tmp_path / "a.onnx"
    right = tmp_path / "b.onnx"
    _export(left, _TinyNet(), output_names=["output0"])
    right.write_bytes(left.read_bytes())
    image = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
    result = compare_onnx(left, right, image=image)
    assert result.ok
    assert result.image == str(image)
    assert result.feed is not None and result.feed.shape == (1, 3, 64, 64)


def test_script_prints_report(tmp_path, capsys):
    left = tmp_path / "a.onnx"
    right = tmp_path / "b.onnx"
    _export(left, _TinyNet(), output_names=["output0"])
    right.write_bytes(left.read_bytes())
    code = _load_script().main([str(left), str(right)])
    report = capsys.readouterr().out
    assert code == 0
    assert "feed  random" in report
    assert "[1, 3, 64, 64] float32" in report
    assert "PASS" in report


def test_script_missing_file(tmp_path, capsys):
    other = tmp_path / "b.onnx"
    _export(other, _TinyNet(), output_names=["output0"])
    code = _load_script().main([str(tmp_path / "no.onnx"), str(other)])
    captured = capsys.readouterr()
    assert code == 2
    assert "找不到参考模型" in captured.err
    assert captured.out == ""
