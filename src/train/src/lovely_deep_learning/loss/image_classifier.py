from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


class ImageClassifierCriterion(nn.Module):
    """多类分类交叉熵，与检测侧 ``DetectionLossYOLOv8`` 相同由 YAML 注入。

    ``forward(preds, *, net_out, net_in=None)``：``preds`` 为 DAGNet 输出列表（取
    ``preds[0]`` 为 ``(N, C)`` logits）；标签来自 ``net_out["class_id"]``。
    ``net_in`` 预留与检测接口对称，当前未参与计算。
    """

    def __init__(
        self,
        weight: torch.Tensor | Sequence[float] | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(list(weight), dtype=torch.float32)
        self._ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=float(label_smoothing),
            reduction=reduction,
        )

    def forward(
        self,
        preds: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
        *,
        net_out: dict[str, Any],
        net_in: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        _ = net_in
        if isinstance(preds, torch.Tensor):
            logits = preds
        else:
            logits = preds[0]
        targets = net_out["class_id"]
        if not targets.dtype.is_floating_point:
            targets = targets.long()
        return self._ce(logits, targets)
