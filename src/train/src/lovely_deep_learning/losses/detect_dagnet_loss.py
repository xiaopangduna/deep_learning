"""
YOLOv8 风格检测损失（DFL + dist2bbox + CIoU + 多标签 BCE），仅依赖 PyTorch。
与 ``lovely_deep_learning.nn.head.Detect`` 训练阶段输出的三层特征图配套使用。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_anchors(
    feats: List[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """各尺度特征图上网格中心坐标（与 stride 同量纲）及逐点 stride。"""
    dtype, device = feats[0].dtype, feats[0].device
    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []
    for i, st in enumerate(strides):
        _, _, h, w = feats[i].shape
        st_f = float(st.item()) if st.numel() == 1 else float(st)
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((gx, gy), -1).reshape(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), st_f, device=device, dtype=dtype)
        )
    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


def _dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """ltrb 距离 → xyxy（与 anchor 同一坐标系，通常为网格单位）。"""
    lt, rb = distance.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), -1)


def _dfl_decode(pred_dist: torch.Tensor, reg_max: int) -> torch.Tensor:
    """(B, A, 4*reg_max) logits → (B, A, 4) 期望 ltrb。"""
    b, a, _c = pred_dist.shape
    p = pred_dist.view(b, a, 4, reg_max).softmax(-1)
    proj = torch.arange(reg_max, device=p.device, dtype=p.dtype).view(1, 1, 1, -1)
    return (p * proj).sum(-1)


def _bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """xyxy 的 Complete IoU，返回 (N,) 或广播形状。"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    w1, h1 = (b1_x2 - b1_x1).clamp(min=0), (b1_y2 - b1_y1).clamp(min=0)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=0), (b2_y2 - b2_y1).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    c2_x = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    c2_y = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = c2_x.pow(2) + c2_y.pow(2) + eps
    rho2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)
    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + v * alpha)


def _xywh_norm_to_xyxy_pixel(
    xywh_norm: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    cx, cy, w, h = xywh_norm.unbind(-1)
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return torch.stack((x1, y1, x2, y2), -1)


class DetectDAGNetLoss(nn.Module):
    """
    将 ``Detect`` 头训练输出（三层 ``[B, no, h, w]``）与 batch 目标对齐后计算标量损失。

    Parameters
    ----------
    detect
        ``Detect`` 模块引用（读取 ``nc``、``reg_max``、``stride``）。
    box_gain, cls_gain, dfl_gain
        与常见 YOLOv8 默认量级相近的增益。
    iou_match_thresh
        锚点与 GT 的最大 IoU 低于该阈值时不视为正样本。
    """

    def __init__(
        self,
        detect: nn.Module,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        iou_match_thresh: float = 0.5,
    ):
        super().__init__()
        self.detect = detect
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.iou_match_thresh = iou_match_thresh

    def forward(self, preds: List[torch.Tensor], batch: dict) -> torch.Tensor:
        if not isinstance(preds, list) or len(preds) != self.detect.nl:
            raise ValueError("preds 须为 Detect 训练输出的特征层列表。")

        img = batch["img"]
        b, _c, ih, iw = img.shape
        device, dtype = img.device, img.dtype

        strides = self.detect.stride.to(device=device, dtype=dtype)
        reg_max = int(self.detect.reg_max)
        nc = int(self.detect.nc)

        pred_distri, pred_scores = torch.cat(
            [xi.view(b, self.detect.no, -1) for xi in preds], 2
        ).split((reg_max * 4, nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = _make_anchors(preds, strides)
        num_anchors = anchor_points.shape[0]
        anchor_points = anchor_points.unsqueeze(0).expand(b, -1, -1)
        stride_tensor = stride_tensor.unsqueeze(0).expand(b, -1, -1)

        pred_ltrb = _dfl_decode(pred_distri, reg_max)
        pred_bboxes = _dist2bbox(pred_ltrb, anchor_points) * stride_tensor

        t_xyxy, t_cls, fg_mask = self._assign_targets(
            batch, b, ih, iw, pred_bboxes.detach(), device, dtype, nc, num_anchors
        )

        loss_box = pred_bboxes.new_tensor(0.0)
        loss_dfl = pred_distri.new_tensor(0.0)
        if fg_mask.any():
            pbox = pred_bboxes[fg_mask]
            gbox = t_xyxy[fg_mask]
            loss_box = (1.0 - _bbox_ciou(pbox, gbox)).mean()

            anc = anchor_points[fg_mask]
            st = stride_tensor[fg_mask]
            gxyxy = gbox
            lt = (anc[..., 0] - gxyxy[..., 0]) / st.squeeze(-1)
            tp = (anc[..., 1] - gxyxy[..., 1]) / st.squeeze(-1)
            rt = (gxyxy[..., 2] - anc[..., 0]) / st.squeeze(-1)
            bt = (gxyxy[..., 3] - anc[..., 1]) / st.squeeze(-1)
            ltrb_t = torch.stack((lt, tp, rt, bt), -1).clamp_(0, reg_max - 1 - 1e-3)
            tgt_idx = ltrb_t.long()
            logp = pred_distri[fg_mask].view(-1, 4, reg_max)
            loss_dfl = F.cross_entropy(
                logp.view(-1, reg_max), tgt_idx.reshape(-1), reduction="mean"
            )

        target_cls = torch.zeros(b, num_anchors, nc, device=device, dtype=dtype)
        if fg_mask.any():
            bi, ai = torch.nonzero(fg_mask, as_tuple=True)
            target_cls[bi, ai, t_cls[fg_mask].long()] = 1.0
        loss_cls = F.binary_cross_entropy_with_logits(pred_scores, target_cls, reduction="mean")

        return (
            self.box_gain * loss_box
            + self.cls_gain * loss_cls
            + self.dfl_gain * loss_dfl
        )

    def _assign_targets(
        self,
        batch: dict,
        b: int,
        ih: int,
        iw: int,
        pred_bboxes: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        nc: int,
        num_anchors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """逐图：每个 GT 选 IoU 最大的锚点；冲突时保留 IoU 更大者。"""
        t_xyxy = torch.zeros(b, num_anchors, 4, device=device, dtype=dtype)
        t_cls = torch.zeros(b, num_anchors, dtype=torch.long, device=device)
        fg_mask = torch.zeros(b, num_anchors, dtype=torch.bool, device=device)

        batch_idx = batch["batch_idx"].long().to(device)
        gt_cls = batch["cls"].long().to(device)
        gt_xywh = batch["bboxes"].to(device=device, dtype=dtype)

        if gt_xywh.numel() == 0:
            return t_xyxy, t_cls, fg_mask

        gt_xyxy = _xywh_norm_to_xyxy_pixel(gt_xywh, ih, iw)

        for img_i in range(b):
            sel = batch_idx == img_i
            if not sel.any():
                continue
            g_cls = gt_cls[sel]
            g_box = gt_xyxy[sel]
            p_box = pred_bboxes[img_i]
            n_g = g_box.shape[0]
            best_iou = torch.full((num_anchors,), -1.0, device=device, dtype=dtype)
            best_gt = torch.full((num_anchors,), -1, device=device, dtype=torch.long)

            for gi in range(n_g):
                gb = g_box[gi : gi + 1].expand(num_anchors, -1)
                ious = _bbox_ciou(p_box, gb)
                mask = ious > best_iou
                best_iou = torch.where(mask, ious, best_iou)
                best_gt = torch.where(mask, torch.full_like(best_gt, gi), best_gt)

            pos = best_iou >= self.iou_match_thresh
            if not pos.any():
                continue
            gi = best_gt[pos]
            fg_mask[img_i, pos] = True
            t_xyxy[img_i, pos] = g_box[gi]
            t_cls[img_i, pos] = g_cls[gi].clamp(0, nc - 1)

        return t_xyxy, t_cls, fg_mask
