"""分类 DAGNet 输出 → softmax / argmax，供 metrics 与回调统一消费。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierPostProcessor(nn.Module):
    """
    将 ``DAGNet`` 分类输出转为固定字典，与 ``YOLOv8PostProcessor.run`` 用法对齐。

    ``run(dag_out)`` 返回键：``logits`` ``(N,C)``、``probs``、``pred_ids``、``pred_conf``。
    ``ImageClassifierMetric.update(stage, post_out, net_out)`` 从 ``post_out`` / ``net_out`` 取张量。
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def run(self, dag_out: tuple[Any, ...] | list[Any] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(dag_out, torch.Tensor):
            logits = dag_out
        else:
            logits = dag_out[0]
        scaled = logits if self.temperature == 1.0 else logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        pred_conf, pred_ids = probs.max(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred_ids": pred_ids,
            "pred_conf": pred_conf,
        }
