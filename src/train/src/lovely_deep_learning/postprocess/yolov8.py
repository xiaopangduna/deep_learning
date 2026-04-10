"""YOLOv8 检测头输出解码与稀疏化（DAGNet 多尺度特征 → raw → NMS / top-k）。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors

from ..dataset.object_detect import postprocess_detections


def _cxcywh_pixels_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
    )


def _cxcywh_pixels_to_xywh_tl(boxes: torch.Tensor) -> torch.Tensor:
    """像素 cxcywh → 左上角 xywh（与 torchmetrics ``box_format='xywh'`` 常见约定一致）。"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    return torch.stack((x1, y1, w, h), dim=-1)


class YOLOv8PostProcessor(nn.Module):
    """
    将 ``DAGNet`` 检测分支输出解码为像素 cxcywh + 类概率的 ``raw``，再可选 ``postprocess_detections``。

    检测头 merge / 通道拆分 / DFL 期望与 :class:`~lovely_deep_learning.loss.object_detect.DetectionLossYOLOv8`
    共用本类下方静态方法；可将模型输出转为 ``MeanAveragePrecision.update`` 所需的 ``preds`` 列表
    （框布局由 ``map_pred_box_format`` 约定，须与 ``metrics.box_format`` 一致）；``targets`` 由
    ``ObjectDetectMetric`` 从 ``net_out`` 组装 ``targets``。另提供 ``(B, max_det, 6)`` 张量供 ``predict`` 等。
    """

    @staticmethod
    def merge_yolov8_head_feats(feats: List[torch.Tensor], no: int) -> torch.Tensor:
        """多层 ``(B, no, H_i, W_i)`` → ``(B, no, A)``，``A = Σ_i H_i W_i``。"""
        batch_size = feats[0].shape[0]
        return torch.cat([xi.view(batch_size, no, -1) for xi in feats], dim=2)

    @staticmethod
    def split_merged_head_reg_cls(
        merged: torch.Tensor, reg_max: int, nc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ``(B, no, A)`` → 回归 ``(B, 4*reg_max, A)`` 与分类 ``(B, nc, A)``（与 ``raw`` 解码同一布局）。
        """
        return merged.split((reg_max * 4, nc), dim=1)

    @staticmethod
    def split_merged_head_to_loss_pred_scores_and_distri(
        merged: torch.Tensor, reg_max: int, nc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ``(B, no, A)`` → ``pred_scores`` ``(B, A, nc)``、``pred_distri`` ``(B, A, 4*reg_max)``（TAL / 损失用）。
        """
        pred_reg, pred_cls = YOLOv8PostProcessor.split_merged_head_reg_cls(
            merged, reg_max, nc
        )
        pred_scores = pred_cls.permute(0, 2, 1).contiguous()
        pred_distri = pred_reg.permute(0, 2, 1).contiguous()
        return pred_scores, pred_distri

    @staticmethod
    def dfl_logits_to_ltrb_b_a4(pred_distri: torch.Tensor, reg_max: int) -> torch.Tensor:
        """
        DFL logits → 期望 ltrb，``(B, A, 4*reg_max)`` → ``(B, A, 4)``；``reg_max <= 1`` 时原样返回。

        与 Ultralytics / ``v8DetectionLoss`` 一致：``softmax`` 后对 bin 索引 ``matmul``。
        """
        if reg_max <= 1:
            return pred_distri
        b, a, _c = pred_distri.shape
        p = pred_distri.view(b, a, 4, reg_max).softmax(-1)
        proj = torch.arange(reg_max, device=pred_distri.device, dtype=torch.float32).to(
            pred_distri.dtype
        )
        return p.matmul(proj)

    @staticmethod
    def feats_to_raw_yolov8(
        feats: List[torch.Tensor],
        nc: int,
        reg_max: int,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        """多尺度 head → 密集 ``raw`` ``(B, 4+nc, A)``（像素 cxcywh + sigmoid cls）。"""
        no = nc + reg_max * 4
        merged = YOLOv8PostProcessor.merge_yolov8_head_feats(feats, no)
        pred_dist_bca, cls_bca = YOLOv8PostProcessor.split_merged_head_reg_cls(
            merged, reg_max, nc
        )
        pred_distri_ba = pred_dist_bca.permute(0, 2, 1).contiguous()
        ltrb_ba4 = YOLOv8PostProcessor.dfl_logits_to_ltrb_b_a4(pred_distri_ba, reg_max)
        pred_dist_b4a = ltrb_ba4.permute(0, 2, 1).contiguous()
        stride = stride.to(device=pred_dist_b4a.device, dtype=pred_dist_b4a.dtype)
        anchors, strides = (
            x.transpose(0, 1) for x in make_anchors(feats, stride, 0.5)
        )
        dbox = dist2bbox(pred_dist_b4a, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        return torch.cat((dbox, cls_bca.sigmoid()), 1)

    def __init__(
        self,
        nc: int,
        reg_max: int,
        stride: Sequence[float],
        max_det: int,
        nms: bool = True,
        nms_iou: float = 0.7,
        inference_conf_thres: float = 0.001,
        map_pred_box_format: str = "xyxy",
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.max_det = int(max_det)
        self.nms = bool(nms)
        self.nms_iou = float(nms_iou)
        self.inference_conf_thres = float(inference_conf_thres)
        self.map_pred_box_format = str(map_pred_box_format).lower()
        self.register_buffer(
            "stride",
            torch.tensor(list(stride), dtype=torch.float32),
            persistent=False,
        )

    def dag_out_to_raw(self, dag_out: tuple) -> torch.Tensor:
        """``DAGNet`` 单输出元组（多尺度特征 list）→ ``raw`` ``(B, 4+nc, A)``（像素 cxcywh + sigmoid cls）。"""
        feats = dag_out[0]
        if not isinstance(feats, list):
            if isinstance(feats, tuple):
                return feats[0]
            return feats
        return YOLOv8PostProcessor.feats_to_raw_yolov8(
            feats, self.nc, self.reg_max, self.stride
        )

    def raw_to_detections(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw`` ``(B, 4+nc, A)`` → ``(B, max_det, 6)``（cxcywh, conf, cls）。"""
        return postprocess_detections(
            raw=raw,
            max_det=self.max_det,
            nc=self.nc,
            nms=self.nms,
            conf_thres=self.inference_conf_thres,
            nms_iou=self.nms_iou,
        )

    def dag_out_to_detections(self, dag_out: tuple) -> torch.Tensor:
        """
        从 **检测头多尺度特征** 一步得到 **稀疏检测结果**（本方法内部无状态，仅串联两步）。

        输入 ``dag_out``
            ``DAGNet`` 前向的 **单输出节点** 包装：``dag_out[0]`` 一般为各尺度的
            ``List[Tensor]``，每层 ``(B, no, H, W)``，``no = 4*reg_max + nc``；也兼容
            ``dag_out[0]`` 已是 ``(B, 4+nc, A)`` 的 ``raw`` 或 ``(raw, feats)`` 元组等少量格式
            （与 :meth:`dag_out_to_raw` 一致）。

        处理流程
            #. :meth:`dag_out_to_raw` — 锚点 + DFL 解码 → 密集 ``raw`` ``(B, 4+nc, A)``（像素 cxcywh + 类分）。
            #. :meth:`raw_to_detections` — conf 阈值、按类 NMS、``max_det`` 截断 → 每图至多
               ``max_det`` 行，每行 ``[cx, cy, w, h, conf, cls]``（像素 cxcywh）。

        返回值
            ``Tensor``，形状 ``(B, max_det, 6)``；不足 ``max_det`` 的位置通常为补零行，
            具体约定见 :func:`~lovely_deep_learning.dataset.object_detect.postprocess_detections`。

        命名说明
            当前名强调 ``DAGNet`` 的 ``tuple`` 包装；若希望与具体网络解耦，可改为侧重
            **「head 特征 → 最终框」** 或 **「decode + postprocess」** 语义的名称（例如
            与 ``forward``、``predict``、``decode`` 等团队约定对齐），再全局替换调用处即可。
        """
        return self.raw_to_detections(self.dag_out_to_raw(dag_out))

    def detections_to_mean_ap_preds(
        self,
        detections: torch.Tensor,
        *,
        conf_thres: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        ``(B, max_det, 6)``（cxcywh, conf, cls）→ ``MeanAveragePrecision.update`` 的 ``preds`` 列表。

        每元素为 ``dict``，含 ``boxes`` ``(N,4)``、``scores`` ``(N,)``、``labels`` ``(N,)`` long。
        ``boxes`` 的布局由 ``map_pred_box_format`` 决定，**须与构造 ``MeanAveragePrecision`` 时的
        ``box_format`` 一致**（例如均为 ``xyxy``）。
        """
        thr = float(self.inference_conf_thres if conf_thres is None else conf_thres)
        bf = self.map_pred_box_format
        preds: list[dict[str, torch.Tensor]] = []
        for det_row in detections:
            pred_mask = det_row[:, 4] > thr
            row = det_row[pred_mask]
            if row.numel() == 0:
                z4 = det_row.new_zeros((0, 4))
                zs = det_row.new_zeros((0,))
                zl = torch.zeros((0,), device=det_row.device, dtype=torch.long)
                preds.append({"boxes": z4, "scores": zs, "labels": zl})
                continue
            cxcywh = row[:, :4].float()
            if bf == "xyxy":
                pred_boxes = _cxcywh_pixels_to_xyxy(cxcywh)
            elif bf in ("cxcywh", "xywh"):
                pred_boxes = (
                    cxcywh if bf == "cxcywh" else _cxcywh_pixels_to_xywh_tl(cxcywh)
                )
            else:
                raise ValueError(
                    f"map_pred_box_format 须为 xyxy / cxcywh / xywh，收到 {bf!r}"
                )
            preds.append(
                {
                    "boxes": pred_boxes,
                    "scores": row[:, 4].float(),
                    "labels": row[:, 5].long(),
                }
            )
        return preds

    def run(
        self,
        dag_out: tuple,
        *,
        conf_thres: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        从 ``DAGNet`` 检测输出得到 **最终预测**（解码 + NMS / top-k），格式为
        ``MeanAveragePrecision.update`` 所需的 ``preds`` 列表。

        同一结果可用于 metrics、可视化、统计等。
        """
        dets = self.dag_out_to_detections(dag_out)
        return self.detections_to_mean_ap_preds(dets, conf_thres=conf_thres)
