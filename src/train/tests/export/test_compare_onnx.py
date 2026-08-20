from __future__ import annotations

import numpy as np
import torch.nn as nn

from lovely_deep_learning.export.base import BaseExporter
from lovely_deep_learning.export.compare import check_io, compare_arrays, main, summarize_onnx
from lovely_deep_learning.postprocess.yolov8 import YOLOv8Decode

from .test_export_graph import _FakeDAGNet


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return (self.conv(x[0]),)


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


def test_check_io_detects_output_count(tmp_path):
    a = tmp_path / "a.onnx"
    b = tmp_path / "b.onnx"
    _export(
        a,
        _FakeDAGNet(),
        output_names=["output0"],
        export_head=YOLOv8Decode(nc=80, reg_max=16, stride=[8, 16, 32]),
    )
    _export(b, _FakeDAGNet())
    errors = check_io(summarize_onnx(a), summarize_onnx(b))
    assert any("输出个数" in e for e in errors)


def test_compare_arrays_allclose():
    ref = np.zeros((1, 84, 8), dtype=np.float32)
    other = ref + 1e-6
    stats = compare_arrays(ref, other, atol=1e-4, rtol=1e-4)
    assert stats["ok"]
    stats_fail = compare_arrays(ref, other + 1.0, atol=1e-4, rtol=1e-4)
    assert not stats_fail["ok"]


def test_compare_onnx_cli_same_graph_passes(tmp_path):
    a = tmp_path / "a.onnx"
    b = tmp_path / "b.onnx"
    _export(a, _TinyNet(), output_names=["output0"])
    b.write_bytes(a.read_bytes())
    assert main([str(a), str(b), "--seed", "0"]) == 0


def test_compare_onnx_cli_missing_file(tmp_path):
    missing = tmp_path / "no.onnx"
    other = tmp_path / "b.onnx"
    _export(other, _TinyNet(), output_names=["output0"])
    assert main([str(missing), str(other)]) == 2
