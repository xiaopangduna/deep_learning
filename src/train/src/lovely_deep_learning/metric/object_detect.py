"""
目标检测验证指标：基于 `TorchMetrics`_ 的
:class:`torchmetrics.detection.MeanAveragePrecision` 计算 COCO 风格 **mAP** / **mAR**
（默认 IoU 0.50–0.95 步长 0.05；与 Lightning 生态一致）。

预测张量格式与 :class:`~lovely_deep_learning.module.object_detect.ObjectDetectModule.run_inference`
一致：``(B, M, 6)`` — ``cx, cy, w, h``（像素）、``score``、``class_id``。
GT 为每样本绝对像素 ``xyxy`` 与 0-based 类别 id。

.. _TorchMetrics: https://lightning.ai/docs/torchmetrics/stable/
"""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection import MeanAveragePrecision

__all__ = ["ObjectDetectMetrics", "batch_to_detection_inputs"]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` cxcywh 像素 → xyxy。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
    )


def batch_to_detection_inputs(
    preds: torch.Tensor,
    gt_boxes_list: list[torch.Tensor],
    gt_classes_list: list[torch.Tensor],
    *,
    conf_thres: float = 1e-6,
    device: torch.device | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """
    将 ``run_inference`` 输出与 GT 转为 TorchMetrics mAP 所需的 ``preds`` / ``target`` 列表
   （每张图一个 ``dict``：``boxes``, ``scores``, ``labels`` / ``boxes``, ``labels``）。

    Args:
        preds: ``(B, M, 6)``
        gt_boxes_list: 长度 ``B``，每项 ``(N, 4)`` xyxy
        gt_classes_list: 长度 ``B``，每项 ``(N,)`` 整型类别（0-based）
        conf_thres: 置信度过滤
        device: 若给定，将张量放到该设备（否则与 ``preds`` 同设备）
    """
    if preds.dim() != 3 or preds.shape[-1] != 6:
        raise ValueError(f"preds 期望 (B, M, 6)，得到 {tuple(preds.shape)}")
    bsz = preds.shape[0]
    if len(gt_boxes_list) != bsz or len(gt_classes_list) != bsz:
        raise ValueError("gt_boxes_list / gt_classes_list 长度须等于 batch 大小")

    dev = device if device is not None else preds.device
    dtype = preds.dtype

    preds_tm: list[dict[str, torch.Tensor]] = []
    target_tm: list[dict[str, torch.Tensor]] = []

    zbox = lambda: torch.zeros(0, 4, device=dev, dtype=dtype)

    for b in range(bsz):
        row = preds[b]
        m = row[:, 4] > conf_thres
        if m.any():
            boxes = _cxcywh_to_xyxy(row[m, :4]).to(device=dev, dtype=dtype)
            preds_tm.append(
                {
                    "boxes": boxes,
                    "scores": row[m, 4].to(device=dev, dtype=dtype),
                    "labels": row[m, 5].long().to(device=dev),
                }
            )
        else:
            preds_tm.append(
                {
                    "boxes": zbox(),
                    "scores": torch.zeros(0, device=dev, dtype=dtype),
                    "labels": torch.zeros(0, dtype=torch.long, device=dev),
                }
            )

        gb = gt_boxes_list[b]
        gc = gt_classes_list[b]
        if gb.numel() > 0:
            target_tm.append(
                {
                    "boxes": gb.to(device=dev, dtype=dtype),
                    "labels": gc.long().to(device=dev),
                }
            )
        else:
            target_tm.append(
                {
                    "boxes": zbox(),
                    "labels": torch.zeros(0, dtype=torch.long, device=dev),
                }
            )

    return preds_tm, target_tm


def _tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def torchmetrics_output_to_report(tm_out: dict[str, Any]) -> dict[str, Any]:
    """将 :meth:`MeanAveragePrecision.compute` 的字典转为易序列化的 Python 标量 / 列表。"""
    report: dict[str, Any] = {}

    def add_float(key_tm: str, key_out: str | None = None) -> None:
        if key_tm not in tm_out:
            return
        t = tm_out[key_tm]
        if not isinstance(t, torch.Tensor):
            return
        v = _tensor_to_float(t) if t.numel() == 1 else t.detach().cpu().tolist()
        report[key_out or key_tm] = v

    add_float("map", "mAP50-95")
    add_float("map_50", "mAP50")
    add_float("map_75", "mAP75")
    add_float("map_small")
    add_float("map_medium")
    add_float("map_large")
    add_float("mar_1")
    add_float("mar_10")
    add_float("mar_100")

    if "map_per_class" in tm_out:
        mpc = tm_out["map_per_class"]
        if isinstance(mpc, torch.Tensor) and mpc.numel() > 0:
            report["mAP50-95_per_class"] = mpc.detach().cpu().tolist()

    if "classes" in tm_out:
        cl = tm_out["classes"]
        if isinstance(cl, torch.Tensor):
            report["classes_observed"] = cl.detach().cpu().tolist()

    return report


class ObjectDetectMetrics:
    """
    封装 `TorchMetrics`_ 的 :class:`~torchmetrics.detection.MeanAveragePrecision`，
    在验证循环中按 batch 调用 :meth:`update`，在 epoch 末调用 :meth:`compute_epoch`。

    参数（构造 ``cfgs`` 字典）：

    - ``iou_thresholds`` (*optional*): 默认 ``None`` 即 COCO 的 0.50–0.95（步长 0.05）
    - ``class_metrics`` (*bool*): 是否返回 ``map_per_class`` 等，默认 ``True``
    - ``backend``: ``\"pycocotools\"`` 或 ``\"faster_coco_eval\"``（需单独安装）
    - ``conf_thres``: 过滤低分预测，默认 ``1e-6``

    主指标：``metric`` 与 ``report['mAP50-95']`` 对齐（与 TorchMetrics 的 ``map`` 一致）。

    .. _TorchMetrics: https://lightning.ai/docs/torchmetrics/stable/
    """

    def __init__(self, cfgs: dict[str, Any]):
        iou = cfgs.get("iou_thresholds")
        if iou is not None:
            iou = list(iou)
        self._conf_thres = float(cfgs.get("conf_thres", 1e-6))
        class_metrics = bool(cfgs.get("class_metrics", True))
        backend = cfgs.get("backend", "pycocotools")

        self._map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=iou,
            class_metrics=class_metrics,
            backend=backend,
        )

        self.report: dict[str, Any] | None = None
        self.metric: float | None = None

    def reset(self) -> None:
        self._map.reset()
        self.report = None
        self.metric = None

    def update(
        self,
        preds: torch.Tensor,
        gt_boxes_list: list[torch.Tensor],
        gt_classes_list: list[torch.Tensor],
    ) -> None:
        """累积一个 batch（内部调用 ``MeanAveragePrecision.update``）。"""
        p, t = batch_to_detection_inputs(
            preds,
            gt_boxes_list,
            gt_classes_list,
            conf_thres=self._conf_thres,
            device=preds.device,
        )
        self._map.update(p, t)

    def compute(self) -> dict[str, Any]:
        """对当前累积数据计算指标（不 ``reset``；可多次调用结果相同）。"""
        tm_out = self._map.compute()
        rep = torchmetrics_output_to_report(tm_out)
        self.report = rep
        self.metric = float(rep.get("mAP50-95", rep.get("mAP50", 0.0)))
        return rep

    def compute_epoch(self) -> dict[str, Any]:
        """计算指标并 ``reset`` 内部状态，便于下一验证 epoch 重新累积。"""
        out = self.compute()
        self._map.reset()
        return out

    # --- 与旧命名兼容（可选） ---
    def reset_metrics(self) -> None:
        self.reset()

    def update_metrics_batch(
        self,
        preds: torch.Tensor,
        gt_boxes_list: list[torch.Tensor],
        gt_classes_list: list[torch.Tensor],
    ) -> None:
        self.update(preds, gt_boxes_list, gt_classes_list)

    def update_metrics_epoch(self) -> None:
        self.compute_epoch()

    def get_report(self) -> dict[str, Any]:
        return dict(self.report) if self.report else {}
