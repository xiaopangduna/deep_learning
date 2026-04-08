from __future__ import annotations

import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ObjectDetectMetric(nn.Module):
    """统一管理目标检测 train/val/test 的 mAP 指标实例。"""

    def __init__(self, box_format: str = "xyxy", iou_type: str = "bbox") -> None:
        super().__init__()
        self._stage_to_key = {
            "train": "train_map",
            "val": "val_map",
            "test": "test_map",
        }
        self._metrics = nn.ModuleDict(
            {
                "train_map": MeanAveragePrecision(box_format=box_format, iou_type=iou_type),
                "val_map": MeanAveragePrecision(box_format=box_format, iou_type=iou_type),
                "test_map": MeanAveragePrecision(box_format=box_format, iou_type=iou_type),
            }
        )

    def update(self, stage: str, preds, targets) -> None:
        self._metrics[self._stage_to_key[stage]].update(preds, targets)

    def compute(self, stage: str):
        return self._metrics[self._stage_to_key[stage]].compute()

    def reset(self, stage: str) -> None:
        self._metrics[self._stage_to_key[stage]].reset()
