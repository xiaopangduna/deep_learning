from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torchmetrics import Accuracy

_AVERAGES = ("micro", "macro")


class ImageClassifierMetric(nn.Module):
    """多类分类 Accuracy：``update(stage, post_out, net_out)``。

    从 ``post_out["logits"]`` 与 ``net_out["class_id"]`` 更新 ``torchmetrics.Accuracy``。
    ``average="micro"`` 是全体样本准确率，``compute`` 的键为 ``acc``。
    ``average="macro"`` 是各类 recall 的平均，键为 ``acc_macro``。
    """

    def __init__(
        self,
        num_classes: int,
        average: Literal["micro", "macro"] = "micro",
    ) -> None:
        super().__init__()
        if average not in _AVERAGES:
            raise ValueError(f"average 须为 micro 或 macro，收到 {average!r}")
        self.average = average
        self._score_key = "acc" if average == "micro" else "acc_macro"
        self._stage_to_key = {
            "train": "train_acc_metric",
            "val": "val_acc_metric",
            "test": "test_acc_metric",
        }
        self._metrics = nn.ModuleDict(
            {
                key: Accuracy(
                    task="multiclass",
                    num_classes=num_classes,
                    average=average,
                )
                for key in self._stage_to_key.values()
            }
        )

    def update(
        self,
        stage: str,
        post_out: dict[str, Any],
        net_out: dict[str, Any],
    ) -> None:
        logits = post_out["logits"]
        targets = net_out["class_id"]
        self._metrics[self._stage_to_key[stage]].update(logits, targets)

    def compute(self, stage: str) -> dict[str, torch.Tensor]:
        key = self._stage_to_key[stage]
        return {self._score_key: self._metrics[key].compute()}

    def reset(self, stage: str) -> None:
        self._metrics[self._stage_to_key[stage]].reset()
