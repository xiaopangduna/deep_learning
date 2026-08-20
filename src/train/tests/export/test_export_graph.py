from __future__ import annotations

import onnx
import torch
import torch.nn as nn

from lovely_deep_learning.export.base import BaseExporter, ExportGraph
from lovely_deep_learning.postprocess.yolov8 import YOLOv8Decode


class _FakeDAGNet(nn.Module):
    """Mimics DAGNet detect output: tuple wrapping a list of 3 feature maps."""

    def forward(self, x):
        images = x[0]
        b = images.shape[0]
        device, dtype = images.device, images.dtype
        feats = [
            torch.zeros(b, 144, 8, 8, device=device, dtype=dtype),
            torch.zeros(b, 144, 4, 4, device=device, dtype=dtype),
            torch.zeros(b, 144, 2, 2, device=device, dtype=dtype),
        ]
        return (feats,)


def test_export_graph_without_head_keeps_three_feats():
    graph = ExportGraph(_FakeDAGNet(), head=None)
    graph.eval()
    with torch.no_grad():
        out = graph(torch.zeros(1, 3, 64, 64))
    assert isinstance(out, tuple)
    feats = out[0]
    assert len(feats) == 3
    assert feats[0].shape == (1, 144, 8, 8)
    assert feats[1].shape == (1, 144, 4, 4)
    assert feats[2].shape == (1, 144, 2, 2)


def test_export_graph_with_decode_head_is_single_raw():
    decode = YOLOv8Decode(nc=80, reg_max=16, stride=[8, 16, 32])
    graph = ExportGraph(_FakeDAGNet(), head=decode)
    graph.eval()
    with torch.no_grad():
        raw = graph(torch.zeros(1, 3, 64, 64))
    anchors = 8 * 8 + 4 * 4 + 2 * 2
    assert raw.shape == (1, 84, anchors)


def test_base_exporter_onnx_with_and_without_head(tmp_path):
    dummy_shape = [1, 3, 64, 64]
    model = _FakeDAGNet()

    raw_path = tmp_path / "raw.onnx"
    BaseExporter(
        onnx_cfg={
            "input_shape": dummy_shape,
            "path_save_onnx": str(raw_path),
            "cfg_for_torch_onnx_export": {
                "export_params": True,
                "opset_version": 12,
                "input_names": ["images"],
            },
        }
    ).export_onnx(model)
    raw_onnx = onnx.load(str(raw_path))
    assert len(raw_onnx.graph.output) == 3

    decoded_path = tmp_path / "decoded.onnx"
    BaseExporter(
        export_head=YOLOv8Decode(nc=80, reg_max=16, stride=[8, 16, 32]),
        onnx_cfg={
            "input_shape": dummy_shape,
            "path_save_onnx": str(decoded_path),
            "cfg_for_torch_onnx_export": {
                "export_params": True,
                "opset_version": 12,
                "input_names": ["images"],
                "output_names": ["output0"],
            },
        },
    ).export_onnx(model)
    decoded = onnx.load(str(decoded_path))
    assert [o.name for o in decoded.graph.output] == ["output0"]
    dims = [d.dim_value for d in decoded.graph.output[0].type.tensor_type.shape.dim]
    assert dims == [1, 84, 8 * 8 + 4 * 4 + 2 * 2]


def _onnx_cfg(path, *, simplify: bool = False) -> dict:
    return {
        "input_shape": [1, 3, 64, 64],
        "path_save_onnx": str(path),
        "simplify": simplify,
        "cfg_for_torch_onnx_export": {
            "export_params": True,
            "opset_version": 13,
            "input_names": ["images"],
            "output_names": ["output0"],
        },
    }


def test_base_exporter_onnx_simplify_keeps_io(tmp_path):
    model = _FakeDAGNet()
    head = YOLOv8Decode(nc=80, reg_max=16, stride=[8, 16, 32])
    raw_path = tmp_path / "raw.onnx"
    slim_path = tmp_path / "slim.onnx"
    BaseExporter(export_head=head, onnx_cfg=_onnx_cfg(raw_path, simplify=False)).export_onnx(
        model
    )
    BaseExporter(export_head=head, onnx_cfg=_onnx_cfg(slim_path, simplify=True)).export_onnx(
        model
    )
    raw = onnx.load(str(raw_path))
    slim = onnx.load(str(slim_path))
    assert [o.name for o in slim.graph.output] == ["output0"]
    raw_dims = [d.dim_value for d in raw.graph.output[0].type.tensor_type.shape.dim]
    slim_dims = [d.dim_value for d in slim.graph.output[0].type.tensor_type.shape.dim]
    assert slim_dims == raw_dims == [1, 84, 8 * 8 + 4 * 4 + 2 * 2]
    assert len(slim.graph.node) <= len(raw.graph.node)


def _transpose_perms(model) -> list[list[int]]:
    return [
        list(a.ints)
        for n in model.graph.node
        if n.op_type == "Transpose"
        for a in n.attribute
        if a.name == "perm"
    ]


def test_decode_onnx_dfl_has_no_3d_transpose(tmp_path):
    """SGS IPU 不能降 3D Transpose。未 simplify 时应有官方 4D DFL perm；slim 可折叠掉 Transpose。"""
    raw_path = tmp_path / "raw.onnx"
    slim_path = tmp_path / "slim.onnx"
    head = YOLOv8Decode(nc=80, reg_max=16, stride=[8, 16, 32])
    BaseExporter(export_head=head, onnx_cfg=_onnx_cfg(raw_path, simplify=False)).export_onnx(
        _FakeDAGNet()
    )
    BaseExporter(export_head=head, onnx_cfg=_onnx_cfg(slim_path, simplify=True)).export_onnx(
        _FakeDAGNet()
    )
    raw_perms = _transpose_perms(onnx.load(str(raw_path)))
    slim_perms = _transpose_perms(onnx.load(str(slim_path)))
    assert [0, 2, 1, 3] in raw_perms
    assert not any(len(p) == 3 for p in raw_perms)
    assert not any(len(p) == 3 for p in slim_perms)
