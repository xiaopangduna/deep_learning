"""YOLOv8 参数分组与线性衰减（对齐 Ultralytics ``build_optimizer`` / ``lf``）。"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def split_yolo_param_groups(
    root: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """``g0`` conv 权重（有 wd）、``g1`` Norm 权重（无 wd）、``g2`` bias（无 wd）。"""
    g0: list[nn.Parameter] = []
    g1: list[nn.Parameter] = []
    g2: list[nn.Parameter] = []
    bn_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for module_name, module in root.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                g2.append(param)
            elif isinstance(module, bn_types):
                g1.append(param)
            else:
                g0.append(param)
    return g0, g1, g2


def yolo_linear_lf(epochs: int, lrf: float) -> Callable[[int], float]:
    """``(1 - epoch/epochs) * (1 - lrf) + lrf``。"""
    n = max(int(epochs), 1)
    r = float(lrf)

    def lf(epoch: int) -> float:
        return (1.0 - float(epoch) / float(n)) * (1.0 - r) + r

    return lf


def interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return float(y1)
    t = (float(x) - float(x0)) / (float(x1) - float(x0))
    t = min(max(t, 0.0), 1.0)
    return float(y0) + t * (float(y1) - float(y0))


def apply_yolo_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    ni: int,
    nw: int,
    epoch: int,
    lf: Callable[[int], float],
    warmup_bias_lr: float,
    warmup_momentum: float,
    momentum: float,
) -> None:
    """``ni<=nw`` 时：pg0 bias 从 ``warmup_bias_lr`` 降到目标 lr，其余从 0 升到目标 lr。"""
    if nw < 0 or ni > nw:
        return
    target_factor = lf(epoch)
    for j, group in enumerate(optimizer.param_groups):
        initial = float(group.get("initial_lr", group["lr"]))
        start = float(warmup_bias_lr) if j == 0 else 0.0
        group["lr"] = interp(float(ni), 0.0, float(nw), start, initial * target_factor)
        if "momentum" in group:
            group["momentum"] = interp(
                float(ni), 0.0, float(nw), float(warmup_momentum), float(momentum)
            )
