from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchmetrics import Accuracy


class ImageClassifierMetric(nn.Module):
    """多类分类 Accuracy：``update(stage, post_out, net_out)``。

    从 ``post_out["logits"]`` 与 ``net_out["class_id"]`` 更新 ``torchmetrics.Accuracy``。
    与检测侧 ``ObjectDetectMetric.update(stage, preds, net_out)`` 形态一致。
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._stage_to_key = {
            "train": "train_acc_metric",
            "val": "val_acc_metric",
            "test": "test_acc_metric",
        }
        self._metrics = nn.ModuleDict(
            {
                "train_acc_metric": Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                "val_acc_metric": Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                "test_acc_metric": Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
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
        return {"acc": self._metrics[key].compute()}

    def reset(self, stage: str) -> None:
        self._metrics[self._stage_to_key[stage]].reset()
