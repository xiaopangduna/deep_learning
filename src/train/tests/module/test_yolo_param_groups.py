"""YOLO 三组参数与 bias warmup。"""

import torch
from torch import nn

from lovely_deep_learning.module.yolo_optim import (
    apply_yolo_warmup,
    interp,
    split_yolo_param_groups,
    yolo_linear_lf,
)


def test_split_bias_conv_bn():
    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=True),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 8, 1, bias=True),
    )
    g0, g1, g2 = split_yolo_param_groups(m)
    assert any(p is m[0].weight for p in g0) and any(p is m[2].weight for p in g0)
    assert any(p is m[1].weight for p in g1)
    assert any(p is m[0].bias for p in g2) and any(p is m[1].bias for p in g2)
    n = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for p in g0 + g1 + g2) == n


def test_linear_lf_start_and_end():
    lf = yolo_linear_lf(500, 0.01)
    assert abs(lf(0) - 1.0) < 1e-9
    assert abs(lf(500) - 0.01) < 1e-9
    assert 0.01 < lf(250) < 1.0


def test_warmup_bias_starts_high_weights_from_zero():
    m = nn.Sequential(nn.Conv2d(3, 4, 1, bias=True), nn.BatchNorm2d(4))
    g0, g1, g2 = split_yolo_param_groups(m)
    opt = torch.optim.SGD(g2, lr=0.01, momentum=0.937, nesterov=True)
    opt.add_param_group({"params": g0, "weight_decay": 5e-4})
    opt.add_param_group({"params": g1, "weight_decay": 0.0})
    for g in opt.param_groups:
        g["initial_lr"] = 0.01
    lf = yolo_linear_lf(500, 0.01)
    apply_yolo_warmup(
        opt,
        ni=0,
        nw=100,
        epoch=0,
        lf=lf,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        momentum=0.937,
    )
    assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-9
    assert abs(opt.param_groups[1]["lr"] - 0.0) < 1e-9
    assert abs(opt.param_groups[2]["lr"] - 0.0) < 1e-9
    apply_yolo_warmup(
        opt,
        ni=100,
        nw=100,
        epoch=0,
        lf=lf,
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        momentum=0.937,
    )
    assert abs(opt.param_groups[0]["lr"] - 0.01) < 1e-9
    assert abs(opt.param_groups[1]["lr"] - 0.01) < 1e-9


def test_interp_endpoints():
    assert interp(0, 0, 10, 0.1, 0.01) == 0.1
    assert abs(interp(10, 0, 10, 0.1, 0.01) - 0.01) < 1e-12
