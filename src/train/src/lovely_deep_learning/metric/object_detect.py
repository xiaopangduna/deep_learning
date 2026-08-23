from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils.metrics import DetMetrics, box_iou

_IOU_THRS = torch.linspace(0.5, 0.95, 10)


def _xywh_tl_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, w, h = boxes.unbind(-1)
    return torch.stack((x1, y1, x1 + w, y1 + h), dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
    )


def _boxes_to_xyxy(boxes: torch.Tensor, box_format: str) -> torch.Tensor:
    """``preds`` 中 ``boxes`` → 像素 xyxy（与 ``box_iou`` / 官方 val 一致）。"""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    bf = box_format.lower()
    if bf == "xyxy":
        return boxes
    if bf == "cxcywh":
        return _cxcywh_to_xyxy(boxes)
    if bf == "xywh":
        return _xywh_tl_to_xyxy(boxes)
    raise ValueError(f"box_format 须为 xyxy / cxcywh / xywh，收到 {bf!r}")


def _unwrap_boxes(boxes: Any) -> torch.Tensor:
    if hasattr(boxes, "data"):
        boxes = boxes.data
    elif hasattr(boxes, "as_tensor"):
        boxes = boxes.as_tensor()
    return boxes


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> np.ndarray:
    """与 ``ultralytics.engine.validator.BaseValidator.match_predictions`` 相同（无 scipy）。"""
    n_pred = int(pred_classes.shape[0])
    correct = np.zeros((n_pred, int(iouv.numel())), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).detach().cpu().numpy()
    for i, threshold in enumerate(iouv.detach().cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


class ObjectDetectMetric(nn.Module):
    """
    目标检测 mAP：``update(stage, preds, net_out)``。

    使用 Ultralytics ``DetMetrics``（与 ``YOLO.val()`` 同一套：NMS 后最多 300 框全部
    进入 PR 曲线）。不要用 torchmetrics ``MeanAveragePrecision(max_detection_thresholds=[1,10,300])``：
    pycocotools 的 ``summarize`` 写死查找 ``maxDets=100``，对不上会得到 ``map=-1``。

    ``preds`` 为检测后处理列表（``boxes`` / ``scores`` / ``labels``）；``net_out`` 为
    collate 后的 GT。``box_format`` 只描述 ``preds["boxes"]``；GT 始终是像素 xyxy。
    ``iou_type`` 保留以兼容 YAML，当前只做 bbox。
    """

    def __init__(
        self,
        box_format: str = "xyxy",
        iou_type: str = "bbox",
        nc: int = 80,
    ) -> None:
        super().__init__()
        if str(iou_type).lower() not in {"bbox", "box"}:
            raise ValueError(f"ObjectDetectMetric 当前只支持 bbox，收到 iou_type={iou_type!r}")
        self._pred_box_format = str(box_format).lower()
        self._stage_to_key = {
            "train": "train_map",
            "val": "val_map",
            "test": "test_map",
        }
        names = {i: str(i) for i in range(int(nc))}
        self._metrics = {
            "train_map": DetMetrics(names=names),
            "val_map": DetMetrics(names=names),
            "test_map": DetMetrics(names=names),
        }
        self.register_buffer("_iouv", _IOU_THRS.clone(), persistent=False)

    def update(
        self,
        stage: str,
        preds: list[dict[str, torch.Tensor]],
        net_out: Any,
    ) -> None:
        metric = self._metrics[self._stage_to_key[stage]]
        targets = self._build_gt_xyxy(net_out, preds)
        iouv = self._iouv
        niou = int(iouv.numel())
        for pred, gt in zip(preds, targets):
            pred_xyxy = _boxes_to_xyxy(pred["boxes"].float(), self._pred_box_format)
            pred_cls = pred["labels"].reshape(-1)
            pred_conf = pred["scores"].reshape(-1)
            gt_xyxy = gt["boxes"].float()
            gt_cls = gt["labels"].reshape(-1)
            n_pred = int(pred_cls.numel())
            n_gt = int(gt_cls.numel())
            if n_pred == 0 or n_gt == 0:
                tp = np.zeros((n_pred, niou), dtype=bool)
            else:
                iou = box_iou(gt_xyxy, pred_xyxy)
                tp = _match_predictions(pred_cls, gt_cls, iou, iouv)
            metric.update_stats(
                {
                    "tp": tp,
                    "conf": pred_conf.detach().cpu().numpy(),
                    "pred_cls": pred_cls.detach().cpu().numpy(),
                    "target_cls": gt_cls.detach().cpu().numpy(),
                    "target_img": np.unique(gt_cls.detach().cpu().numpy())
                    if n_gt
                    else np.zeros((0,), dtype=np.int64),
                }
            )

    def _build_gt_xyxy(
        self,
        net_out: Any,
        preds: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = len(preds)
        if isinstance(net_out, dict):
            net_out_list = (
                [{k: net_out[k][i] for k in net_out} for i in range(batch_size)]
                if net_out
                else [{} for _ in range(batch_size)]
            )
        else:
            net_out_list = list(net_out)

        if len(net_out_list) != batch_size:
            raise ValueError(
                f"net_out 样本数 {len(net_out_list)} 与 preds 长度 {batch_size} 不一致"
            )

        targets: list[dict[str, torch.Tensor]] = []
        for i, gt in enumerate(net_out_list):
            device = preds[i]["boxes"].device
            if gt and "bboxes_xyxy_abs_tv_transformed" in gt:
                gt_boxes = _unwrap_boxes(gt["bboxes_xyxy_abs_tv_transformed"])
                gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
                gt_labels = gt["cls_tv_transformed"].to(device=device).long().reshape(-1)
            else:
                gt_boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
                gt_labels = torch.zeros((0,), device=device, dtype=torch.long)
            targets.append({"boxes": gt_boxes, "labels": gt_labels})
        return targets

    def compute(self, stage: str) -> dict[str, torch.Tensor]:
        metric = self._metrics[self._stage_to_key[stage]]
        if not metric.stats["tp"]:
            z = torch.tensor(0.0)
            return {"map": z, "map_50": z.clone(), "map_75": z.clone()}
        metric.process(plot=False)
        return {
            "map": torch.tensor(float(metric.box.map)),
            "map_50": torch.tensor(float(metric.box.map50)),
            "map_75": torch.tensor(float(metric.box.map75)),
        }

    def reset(self, stage: str) -> None:
        self._metrics[self._stage_to_key[stage]].clear_stats()
