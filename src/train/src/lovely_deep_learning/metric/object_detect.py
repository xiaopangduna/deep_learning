from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return torch.stack((cx, cy, w, h), dim=-1)


def _xyxy_to_xywh_tl(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)


def _collate_gt_xyxy_to_map_boxes(
    gt_xyxy: torch.Tensor, box_format: str
) -> torch.Tensor:
    """collate 中 GT 为像素 xyxy → 与 ``MeanAveragePrecision(box_format=...)`` 一致。"""
    bf = box_format.lower()
    if bf == "xyxy":
        return gt_xyxy
    if bf == "cxcywh":
        return _xyxy_to_cxcywh(gt_xyxy)
    if bf == "xywh":
        return _xyxy_to_xywh_tl(gt_xyxy)
    raise ValueError(f"box_format 须为 xyxy / cxcywh / xywh，收到 {bf!r}")


class ObjectDetectMetric(nn.Module):
    """
    目标检测 mAP：``update(stage, preds, net_out)``。

    ``preds`` 为 ``torchmetrics`` 约定的预测列表（通常由后处理从模型输出得到）；
    ``net_out`` 为 batch 中每图 GT 的 collate 结构，在此转为 ``targets`` 后写入
    ``MeanAveragePrecision``。``box_format`` 须与后处理中预测框格式一致。
    """

    def __init__(self, box_format: str = "xyxy", iou_type: str = "bbox") -> None:
        super().__init__()
        self._target_box_format = str(box_format).lower()
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

    def update(
        self,
        stage: str,
        preds: list[dict[str, torch.Tensor]],
        net_out: Any,
    ) -> None:
        targets = self._build_map_targets(net_out, preds)
        self._metrics[self._stage_to_key[stage]].update(preds, targets)

    def _build_map_targets(
        self,
        net_out: Any,
        preds: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = len(preds)
        targets: list[dict[str, torch.Tensor]] = []

        if isinstance(net_out, dict):
            net_out_list = (
                [{k: net_out[k][i] for k in net_out}
                    for i in range(batch_size)]
                if net_out
                else [{} for _ in range(batch_size)]
            )
        else:
            net_out_list = list(net_out)

        if len(net_out_list) != batch_size:
            raise ValueError(
                f"net_out 样本数 {len(net_out_list)} 与 preds 长度 {batch_size} 不一致"
            )

        for i, gt in enumerate(net_out_list):
            device = preds[i]["boxes"].device
            if gt and "bboxes_xyxy_abs_tv_transformed" in gt:
                gt_boxes = gt["bboxes_xyxy_abs_tv_transformed"]
                if hasattr(gt_boxes, "data"):
                    gt_boxes = gt_boxes.data
                elif hasattr(gt_boxes, "as_tensor"):
                    gt_boxes = gt_boxes.as_tensor()
                gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
                gt_boxes = _collate_gt_xyxy_to_map_boxes(
                    gt_boxes, self._target_box_format
                )
                gt_labels = gt["cls_tv_transformed"].to(
                    device=device).long().reshape(-1)
            else:
                gt_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                gt_labels = torch.zeros((0,), device=device, dtype=torch.long)

            targets.append({"boxes": gt_boxes, "labels": gt_labels})

        return targets

    def compute(self, stage: str):
        return self._metrics[self._stage_to_key[stage]].compute()

    def reset(self, stage: str) -> None:
        self._metrics[self._stage_to_key[stage]].reset()
