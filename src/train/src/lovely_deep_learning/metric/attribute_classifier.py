from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class AttributeClassifierMetric(nn.Module):
    """各属性准确率的平均。``compute`` 的键是 ``acc``，模块记成 ``val_acc``。

    准确率按属性分别计算再平均，每个属性权重相同。预测来自 ``post_out["pred"]``，
    标签来自 ``net_out["attr"]``。
    """

    def __init__(self, num_attributes: int) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError(f"num_attributes 须为正数，收到 {num_attributes}")
        self.num_attributes = int(num_attributes)
        self._stages = ("train", "val", "test")
        self._correct = {stage: torch.zeros(self.num_attributes) for stage in self._stages}
        self._count = {stage: torch.zeros(()) for stage in self._stages}

    def update(self, stage: str, post_out: dict[str, Any], net_out: dict[str, Any]) -> None:
        pred = self._as_binary(post_out["pred"])
        target = self._as_binary(net_out["attr"])
        if pred.shape != target.shape or pred.shape[1] != self.num_attributes:
            raise ValueError(
                f"预测 {tuple(pred.shape)} 与标签 {tuple(target.shape)} 须为 (N, {self.num_attributes})"
            )
        self._correct[stage] += pred.eq(target).sum(dim=0)
        self._count[stage] += pred.shape[0]

    def compute(self, stage: str) -> dict[str, torch.Tensor]:
        count = self._count[stage]
        if int(count) == 0:
            return {"acc": torch.zeros(())}
        return {"acc": (self._correct[stage] / count).mean()}

    def reset(self, stage: str) -> None:
        self._correct[stage].zero_()
        self._count[stage].zero_()

    @staticmethod
    def _as_binary(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        if value.dtype == torch.bool:
            return value
        return value.ge(0.5)
