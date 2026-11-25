"""属性 DAGNet 输出 → sigmoid / 阈值，供 metrics 消费。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class AttributeClassifierPostProcessor(nn.Module):
    """``run`` 返回 ``logits``、``probs`` 和 ``pred``，形状都是 ``(N, K)``。

    ``pred`` 为 0/1，``probs >= threshold`` 的位置是 1。
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"threshold 须在 (0, 1)，收到 {threshold}")
        self.threshold = float(threshold)

    def run(self, dag_out: tuple[Any, ...] | list[Any] | torch.Tensor) -> dict[str, torch.Tensor]:
        logits = dag_out if isinstance(dag_out, torch.Tensor) else dag_out[0]
        probs = torch.sigmoid(logits)
        pred = probs.ge(self.threshold).to(dtype=probs.dtype)
        return {"logits": logits, "probs": probs, "pred": pred}
