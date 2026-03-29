"""
YOLOv8 检测损失（自研实现，与 Ultralytics ``v8DetectionLoss`` 前向公式对齐，便于对照论文与源码阅读）。

**训练管线（先分配，再算三项损失）**

1. **公共准备**：多尺度特征拼成「全锚点」预测；``make_anchors``；GT 从归一化 xywh 转为像素 xyxy；
   DFL  logits 经 softmax×proj 得到 ltrb，再 ``dist2bbox`` 得到预测框（网格单位 → 乘 stride 与 GT 对齐）。

2. **TaskAlignedAssigner（TAL）**：根据分类得分与 IoU 为每个锚点分配 soft 目标 ``target_scores``、
   回归目标 ``target_bboxes``（像素 xyxy）及前景掩码 ``fg_mask``。三项损失**都依赖**这一步的输出。

3. **三项标量损失（未乘 ``hyp.*`` 前）**

   - **cls（分类）**：对 **所有**锚点做 ``BCEWithLogits(pred_scores, target_scores)``，
     ``target_scores`` 为 TAL 给出的多类 soft label（含大量背景 0），按 ``target_scores_sum`` 归一化。

   - **box（框 / CIoU）**：仅 **fg_mask** 上的锚点；``1 - CIoU(pred, target)``，加权后与 ``target_scores_sum`` 归一化。

   - **dfl（分布焦点）**：仅 **fg_mask**；把 GT 框转为相对锚点的 ltrb 真值，对每条边的 ``reg_max`` 维
     logits 做交叉熵（Ultralytics 的 ``DFLoss``）。``reg_max==1`` 时此项为 0。

最后 ``loss[0,1,2]`` 分别乘 ``hyp.box / hyp.cls / hyp.dfl``，再乘 ``batch_size`` 与官方一致。

不调用 ``v8DetectionLoss`` 类；``TaskAlignedAssigner`` / ``make_anchors`` / ``dist2bbox`` / ``bbox2dist``
仍使用 Ultralytics 工具实现。辅助函数 ``_make_anchors``、``_dfl_decode`` 等保留供单测。
"""

from __future__ import annotations

import copy
from typing import Any, List, NamedTuple, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors


def merge_yolov8_hyp_args(
    box_gain: float | None,
    cls_gain: float | None,
    dfl_gain: float | None,
) -> Any:
    """``DEFAULT_CFG`` 拷贝，可选覆盖 box/cls/dfl（与 Ultralytics 训练超参名一致）。"""
    args = copy.deepcopy(DEFAULT_CFG)
    if box_gain is not None:
        args.box = float(box_gain)
    if cls_gain is not None:
        args.cls = float(cls_gain)
    if dfl_gain is not None:
        args.dfl = float(dfl_gain)
    return args


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


class _DFLoss(nn.Module):
    """与 Ultralytics ``DFLoss`` 一致（分布焦点损失）。"""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class _BboxLoss(nn.Module):
    """与 Ultralytics ``BboxLoss`` 一致：CIoU + DFL。"""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = _DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def _preprocess_v8_targets(
    targets: torch.Tensor,
    batch_size: int,
    scale_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """与 ``v8DetectionLoss.preprocess`` 一致：按 batch 铺平 GT，xywh→xyxy（像素）。"""
    nl, ne = targets.shape
    if nl == 0:
        return torch.zeros(batch_size, 0, ne - 1, device=device)
    i = targets[:, 0]
    _, counts = i.unique(return_counts=True)
    counts = counts.to(dtype=torch.int32)
    out = torch.zeros(batch_size, counts.max(), ne - 1, device=device)
    for j in range(batch_size):
        matches = i == j
        if n := matches.sum():
            out[j, :n] = targets[matches, 1:]
    out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out


def _as_stride_tensor(stride: Union[torch.Tensor, Sequence[float]]) -> torch.Tensor:
    """各尺度 stride，形状 ``(nl,)``，与 Detect 头一致（如 ``[8, 16, 32]``）。"""
    if isinstance(stride, torch.Tensor):
        return stride.detach().clone().float().flatten()
    return torch.tensor(list(stride), dtype=torch.float32)


class _TALAssignOut(NamedTuple):
    """TAL 分配结果，供 cls / box / dfl 三项共用。"""

    target_bboxes_px: torch.Tensor
    target_scores: torch.Tensor
    fg_mask: torch.Tensor


class DetectDAGNetLoss:
    """
    YOLOv8 检测损失（自研前向，与 ``v8DetectionLoss`` 数学路径一致）。

    结构：``forward_loss_vec`` → 解析特征 → 锚点与 GT → 任务对齐分配 →
    ``_loss_cls`` / ``_loss_box_dfl`` → 乘 ``hyp`` 与 ``batch_size``。

    只依赖 **Detect 头** 的 ``nc`` / ``reg_max`` / ``stride``，不持有整网。

    Parameters
    ----------
    nc
        类别数（与 Detect 的 ``nc`` 一致）。
    reg_max
        DFL 分布长度（与 Detect 的 ``reg_max`` 一致；为 1 表示不用 DFL）。
    stride
        各检测层步长，长度与特征层数相同（如 ``tensor([8.,16.,32.])`` 或 ``[8, 16, 32]``）。
    box_gain, cls_gain, dfl_gain
        与 Ultralytics ``hyp.box`` / ``hyp.cls`` / ``hyp.dfl`` 一致，``None`` 则用 ``DEFAULT_CFG``。
    tal_topk
        ``TaskAlignedAssigner`` 的 ``topk``。
    """

    def __init__(
        self,
        nc: int,
        reg_max: int,
        stride: Union[torch.Tensor, Sequence[float]],
        box_gain: float | None = None,
        cls_gain: float | None = None,
        dfl_gain: float | None = None,
        tal_topk: int = 10,
        **kwargs: Any,
    ):
        _ = kwargs
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.stride = _as_stride_tensor(stride)
        self.no = self.nc + self.reg_max * 4
        self.use_dfl = self.reg_max > 1

        self.hyp = merge_yolov8_hyp_args(box_gain, cls_gain, dfl_gain)
        self._tal_topk = tal_topk

        self._bce = nn.BCEWithLogitsLoss(reduction="none")
        self._assigner: TaskAlignedAssigner | None = None
        self._bbox_loss: _BboxLoss | None = None
        self._proj: torch.Tensor | None = None
        self._init_device: torch.device | None = None

    def _ensure_heads(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._init_device != device or self._assigner is None:
            self._assigner = TaskAlignedAssigner(
                topk=self._tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
            )
            self._bbox_loss = _BboxLoss(self.reg_max).to(device)
            self._proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
            self._init_device = device

    def _stride_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.stride.to(device=device, dtype=dtype)

    @staticmethod
    def _feature_list(preds: Union[List[torch.Tensor], Tuple[Any, ...]]) -> List[torch.Tensor]:
        return preds[1] if isinstance(preds, tuple) else preds

    def _split_head_outputs(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """三层特征 → ``(B, A, 4*reg_max)`` 与 ``(B, A, nc)``。"""
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        return pred_distri, pred_scores

    def _decode_pred_boxes(
        self, anchor_points: torch.Tensor, pred_distri: torch.Tensor
    ) -> torch.Tensor:
        """DFL 解码 + ``dist2bbox`` → 网格坐标系 xyxy。"""
        if self.use_dfl:
            b, a, c = pred_distri.shape
            pred_distri = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(self._proj.type(pred_distri.dtype))
        return dist2bbox(pred_distri, anchor_points, xywh=False)

    def _ground_truth_tensors(
        self,
        batch: dict,
        batch_size: int,
        imgsz_hw: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``batch`` → 按图铺开的 **像素 xyxy** GT 与 ``mask_gt``。"""
        raw = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = _preprocess_v8_targets(
            raw.to(device), batch_size, scale_tensor=imgsz_hw[[1, 0, 1, 0]], device=device
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        return gt_labels, gt_bboxes, mask_gt

    def _task_aligned_assign(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> _TALAssignOut:
        """TAL：在 **像素空间** 比较预测框与 GT（不读 DFL logits）。"""
        assert self._assigner is not None
        _, target_bboxes_px, target_scores, fg_mask, _ = self._assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        return _TALAssignOut(
            target_bboxes_px=target_bboxes_px,
            target_scores=target_scores,
            fg_mask=fg_mask,
        )

    def _loss_cls(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """分类 BCE：全锚点，目标为 TAL 的 soft 多类标签。"""
        return self._bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

    def _loss_box_dfl(
        self,
        pred_distri: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        tal: _TALAssignOut,
        target_scores_sum: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """框 CIoU + DFL：仅 ``fg_mask``；回归目标在 **网格单位**（除以 stride）。"""
        assert self._bbox_loss is not None
        if not tal.fg_mask.sum():
            z = torch.zeros((), device=pred_distri.device)
            return z, z
        target_bboxes_grid = tal.target_bboxes_px / stride_tensor
        return self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes_grid,
            tal.target_scores,
            target_scores_sum,
            tal.fg_mask,
        )

    def forward_loss_vec(self, preds: List[torch.Tensor], batch: dict) -> torch.Tensor:
        """
        返回 ``(3,)`` 的加权损失：**[box, cls, dfl]**，已乘 ``hyp.*``，再乘 ``batch_size`` 与 Ultralytics 一致。
        """
        feats = self._feature_list(preds)
        device = feats[0].device
        dtype = feats[0].dtype
        self._ensure_heads(device, dtype)
        stride = self._stride_on(device, dtype)

        pred_distri, pred_scores = self._split_head_outputs(feats)
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        gt_labels, gt_bboxes, mask_gt = self._ground_truth_tensors(batch, batch_size, imgsz, device)
        pred_bboxes = self._decode_pred_boxes(anchor_points, pred_distri)

        tal = self._task_aligned_assign(
            pred_scores,
            pred_bboxes,
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(tal.target_scores.sum(), 1)

        l_cls = self._loss_cls(pred_scores, tal.target_scores, target_scores_sum, dtype)
        l_box, l_dfl = self._loss_box_dfl(
            pred_distri,
            pred_bboxes,
            anchor_points,
            tal,
            target_scores_sum,
            stride_tensor,
        )

        loss = torch.zeros(3, device=device)
        loss[0] = l_box * self.hyp.box
        loss[1] = l_cls * self.hyp.cls
        loss[2] = l_dfl * self.hyp.dfl
        return loss * batch_size

    def __call__(self, preds: List[torch.Tensor], batch: dict) -> torch.Tensor:
        if not isinstance(preds, list) and not isinstance(preds, tuple):
            raise ValueError("preds 须为 Detect 训练输出的特征层列表（或含 preds 的 tuple）。")
        return self.forward_loss_vec(preds, batch).sum()


# 测试与旧代码兼容别名
_merge_hyp_args = merge_yolov8_hyp_args


class _V8LossAdapter:
    """仅用于单测：与 Ultralytics ``v8DetectionLoss(model)`` 的 ``model`` 约定一致。"""

    def __init__(self, dagnet: nn.Module, args: Any):
        self._dagnet = dagnet
        last = dagnet.layers_config[-1]["name"]
        self.model = [dagnet.layers[last]]
        self.args = args

    def parameters(self):
        return self._dagnet.parameters()
