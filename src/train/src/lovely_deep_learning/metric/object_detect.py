from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics.utils.tal import dist2bbox, make_anchors

from ..dataset.object_detect import postprocess_detections


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


class ObjectDetectYOLOMetric(ObjectDetectMetric):
    """
    在基类 mAP 之上封装 ``postprocess_detections`` 与 mAP 所需 preds/targets 组装。

    ``Detect._inference`` 得到的 ``raw`` 张量由 ``ObjectDetectModule`` 计算后传入
    :meth:`update`；本类仅依赖 ``init_args`` 中的 ``nc`` / ``max_det`` 与后处理超参，
    不持有 ``Detect`` 模块引用。
    """

    def __init__(
        self,
        nc: int,
        reg_max: int,
        stride: Sequence[float],
        max_det: int,
        nms: bool = True,
        nms_iou: float = 0.7,
        inference_conf_thres: float = 0.001,
        box_format: str = "xyxy",
        iou_type: str = "bbox",
    ) -> None:
        super().__init__(box_format=box_format, iou_type=iou_type)
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.max_det = int(max_det)
        self.nms = bool(nms)
        self.nms_iou = float(nms_iou)
        self.inference_conf_thres = float(inference_conf_thres)
        self.register_buffer(
            "stride",
            torch.tensor(list(stride), dtype=torch.float32),
            persistent=False,
        )

    def _raw_from_dag_out(self, dag_out: tuple) -> torch.Tensor:
        """``DAGNet`` 单输出元组（固定为多尺度特征 list）→ ``raw`` (B, 4+nc, A)。"""
        feats = dag_out[0]
        if not isinstance(feats, list):
            if isinstance(feats, tuple):
                # 兼容少量旧格式：(raw, feats)
                return feats[0]
            return feats
        shape = feats[0].shape
        no = self.nc + self.reg_max * 4
        x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in feats], 2)
        pred_dist, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        if self.reg_max > 1:
            b, _, a = pred_dist.shape
            pred_dist = pred_dist.view(b, 4, self.reg_max, a).softmax(2)
            proj = torch.arange(
                self.reg_max, device=pred_dist.device, dtype=pred_dist.dtype
            ).view(1, 1, self.reg_max, 1)
            pred_dist = (pred_dist * proj).sum(2)
        stride = self.stride.to(device=pred_dist.device, dtype=pred_dist.dtype)
        anchors, strides = (
            x.transpose(0, 1) for x in make_anchors(feats, stride, 0.5)
        )
        dbox = dist2bbox(pred_dist, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def raw_to_detections(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw``：``Detect._inference`` 输出 ``(B, 4+nc, A)`` → ``(B, max_det, 6)``。"""
        return postprocess_detections(
            raw=raw,
            max_det=self.max_det,
            nc=self.nc,
            nms=self.nms,
            conf_thres=self.inference_conf_thres,
            nms_iou=self.nms_iou,
        )

    def to_detections(self, dag_out: tuple) -> torch.Tensor:
        """``DAGNet`` 输出元组直接转检测结果 ``(B, max_det, 6)``。"""
        return self.raw_to_detections(self._raw_from_dag_out(dag_out))

    def update(
        self,
        stage: str,
        dag_out: tuple,
        net_out: Any,
        *,
        detections: Optional[torch.Tensor] = None,
    ) -> None:
        raw = self._raw_from_dag_out(dag_out)
        if detections is None:
            detections = self.raw_to_detections(raw)
        preds, targets = self._build_map_inputs(detections, net_out)
        super().update(stage, preds, targets)

    def _build_map_inputs(
        self, detections: torch.Tensor, net_out: Any
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        preds: list[dict[str, torch.Tensor]] = []
        targets: list[dict[str, torch.Tensor]] = []
        conf_thres = self.inference_conf_thres

        if isinstance(net_out, dict):
            batch_size = detections.shape[0]
            net_out_list = (
                [{k: net_out[k][i] for k in net_out}
                    for i in range(batch_size)]
                if net_out
                else [{} for _ in range(batch_size)]
            )
        else:
            net_out_list = list(net_out)

        for det_row, gt in zip(detections, net_out_list):
            pred_mask = det_row[:, 4] > conf_thres
            pred_row = det_row[pred_mask]
            if pred_row.numel() == 0:
                pred_boxes = det_row.new_zeros((0, 4))
                pred_scores = det_row.new_zeros((0,))
                pred_labels = torch.zeros(
                    (0,), device=det_row.device, dtype=torch.long)
            else:
                pred_boxes = _cxcywh_pixels_to_xyxy(
                    pred_row[:, :4]).float()
                pred_scores = pred_row[:, 4].float()
                pred_labels = pred_row[:, 5].long()

            preds.append(
                {"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}
            )

            if gt and "bboxes_xyxy_abs_tv_transformed" in gt:
                gt_boxes = gt["bboxes_xyxy_abs_tv_transformed"]
                if hasattr(gt_boxes, "data"):
                    gt_boxes = gt_boxes.data
                elif hasattr(gt_boxes, "as_tensor"):
                    gt_boxes = gt_boxes.as_tensor()
                gt_boxes = gt_boxes.to(
                    device=det_row.device, dtype=torch.float32)
                gt_labels = gt["cls_tv_transformed"].to(
                    device=det_row.device).long().reshape(-1)
            else:
                gt_boxes = det_row.new_zeros((0, 4), dtype=torch.float32)
                gt_labels = torch.zeros(
                    (0,), device=det_row.device, dtype=torch.long)

            targets.append({"boxes": gt_boxes, "labels": gt_labels})

        return preds, targets


def _cxcywh_pixels_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
    )
