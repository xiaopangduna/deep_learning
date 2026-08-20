from __future__ import annotations

import torch

from lovely_deep_learning.nn.block import DFL
from lovely_deep_learning.postprocess.yolov8 import YOLOv8Decode, YOLOv8PostProcessor


def _fake_head_feats(batch: int = 2, nc: int = 80, reg_max: int = 16):
    no = nc + reg_max * 4
    torch.manual_seed(0)
    return [
        torch.randn(batch, no, 80, 80),
        torch.randn(batch, no, 40, 40),
        torch.randn(batch, no, 20, 20),
    ]


def test_yolov8_decode_matches_static_feats_to_raw():
    nc, reg_max = 80, 16
    stride = [8, 16, 32]
    feats = _fake_head_feats()
    decode = YOLOv8Decode(nc=nc, reg_max=reg_max, stride=stride)
    decode.eval()
    with torch.no_grad():
        out = decode(feats)
        ref = YOLOv8PostProcessor.feats_to_raw_yolov8(
            feats, nc, reg_max, decode.stride
        )
    assert out.shape == (2, 4 + nc, 80 * 80 + 40 * 40 + 20 * 20)
    assert torch.equal(out, ref)


def test_postprocessor_dag_out_to_raw_uses_decode():
    nc, reg_max = 80, 16
    feats = _fake_head_feats(batch=1)
    pp = YOLOv8PostProcessor(
        nc=nc, reg_max=reg_max, stride=[8, 16, 32], max_det=300, nms=False
    )
    pp.eval()
    with torch.no_grad():
        raw = pp.dag_out_to_raw((feats,))
        via_decode = pp.decode(feats)
    assert torch.equal(raw, via_decode)


def test_dfl_conv_matches_ba_matmul():
    """导出用 4D Conv DFL，与损失用的 (B,A,64) softmax+matmul 数值一致。"""
    torch.manual_seed(0)
    b, a, reg_max = 2, 100, 16
    box_bca = torch.randn(b, 4 * reg_max, a)
    conv_out = DFL(reg_max)(box_bca)
    matmul_out = YOLOv8PostProcessor.dfl_logits_to_ltrb_b_a4(
        box_bca.permute(0, 2, 1).contiguous(), reg_max
    ).permute(0, 2, 1)
    assert torch.allclose(conv_out, matmul_out, atol=1e-5, rtol=1e-5)
