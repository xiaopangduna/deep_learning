"""
YOLOv8 检测损失（自研前向，与 Ultralytics ``v8DetectionLoss`` 对齐）。

**流程概要**

1. **预测与锚点**：多尺度 Detect 输出合并为 ``(B, no, A)``，拆成分类 / 回归；生成 ``(A,2)`` 锚点与 ``(A,1)`` stride；
   DFL 解码 ltrb 后经 ``_dist2bbox`` 得 **网格坐标** xyxy（非 0~1 归一化，需乘 stride 得像素）。

2. **GT**：展平 ``(N,6)`` → 按图 ``(B,n_max,5)`` → 归一化 xywh 转 **像素 xyxy**（``_xywh2xyxy``）。

3. **TAL**：``TaskAlignedAssigner`` 在像素空间分配 ``target_scores``、``target_bboxes``、``fg_mask``。

4. **损失**：cls（全锚点 BCE）+ box（fg 上 CIoU）+ dfl（fg 上分布损失）；再乘 ``hyp`` 与 ``batch_size``。

**依赖**：Ultralytics 的 ``TaskAlignedAssigner``、``bbox2dist``、``bbox_iou``、``DEFAULT_CFG``。
**自实现**：锚点 ``build_flat_anchor_points_and_strides_from_multiscale_feats``、``_xywh2xyxy``、``_dist2bbox``、
``_dfl_decode``（单测对照用）等。
"""

from __future__ import annotations

import copy
from typing import Any, List, NamedTuple, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist


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


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    中心格式 ``(cx, cy, w, h)`` → 角点格式 ``(x1, y1, x2, y2)``，与 Ultralytics
    ``ultralytics.utils.ops.xywh2xyxy`` 一致（最后一维须为 4）。
    """
    assert x.shape[-1] == 4, f"expected last dim 4, got {x.shape}"
    y = torch.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] * 0.5
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def build_flat_anchor_points_and_strides_from_multiscale_feats(
    feats: List[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    多尺度特征上生成网格中心锚点及逐点 stride（与 ``ultralytics.utils.tal.make_anchors`` 在
    ``feats`` 为特征张量列表时行为一致；不依赖 Ultralytics 实现）。

    坐标含义：锚点位于 **该层特征图网格坐标系**（单位：格），中心为 ``(j+offset, i+offset)``；
    与 ``_dist2bbox`` 输出的框同属一格坐标系，再乘 ``stride`` 才到像素。

    Parameters
    ----------
    feats
        每层 ``(B, C, H_i, W_i)``，仅用 ``H_i, W_i`` 决定该层锚点个数与布局。
    strides
        长度与 ``feats`` 相同，第 ``i`` 层下采样步长（如 8/16/32）。
    grid_cell_offset
        网格线偏移，默认 0.5 表示取 **格子中心** 而非左上角。

    Returns
    -------
    anchor_points
        ``(sum_i H_i*W_i, 2)``，每行 ``(x, y)`` 为网格坐标下的锚点中心。
    stride_tensor
        ``(sum_i H_i*W_i, 1)``，与 ``anchor_points`` 逐行对应，该锚点所属层的 stride。
    """
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
        stride_tensor.append(torch.full((h * w, 1), st_f, device=device, dtype=dtype))

    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


def _dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    *,
    xywh: bool = False,
) -> torch.Tensor:
    """
    将 ltrb（相对锚点左/上/右/下距离）转为框；与 ``ultralytics.utils.tal.dist2bbox`` 一致。

    ``distance`` 最后一维为 4：``(lt, top, rb, bottom)`` 语义与 Ultralytics 相同。
    ``anchor_points`` 与 ``distance`` 在 batch/锚点维可广播（常见 ``distance``: ``(B,A,4)``，
    ``anchor_points``: ``(A,2)``）。

    Parameters
    ----------
    xywh
        ``False``（默认）返回 ``xyxy``；``True`` 返回 ``xywh``（中心+宽高），与官方一致。
    """
    lt, rb = distance.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) * 0.5
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), -1)
    return torch.cat((x1y1, x2y2), -1)


def _dfl_decode(pred_dist: torch.Tensor, reg_max: int) -> torch.Tensor:
    """DFL：``(B,A,4*reg_max)`` softmax×bin 索引 → ``(B,A,4)`` 期望 ltrb（单测用）。"""
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
        # target_scores: (B, A, nc) -> 每个锚点的前景权重，形状约为 (#fg, 1)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 仅在前景锚点上计算 CIoU，pred/target 均为 xyxy（同一坐标系）
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # CIoU 损失：1 - iou，并按前景权重加权，再用 target_scores_sum 做归一化
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            # 将目标框（xyxy）转为相对锚点的 ltrb 离散回归目标，范围截断到 reg_max-1
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # pred_dist[fg_mask]: (#fg, 4*reg_max) -> view 后每条边一个 reg_max 分类
            # _DFLoss 输出 (#fg, 1)，再乘同一套前景权重
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            )
            # DFL 同样按 target_scores_sum 归一化，保证与 cls/box 标度一致
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # reg_max==1 时不启用 DFL，保持返回标量张量
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


def group_flat_v8_rows_per_image_by_batch_id(
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将展平行表 ``(N, ne)``（首列为 ``batch_idx``）按图铺开为 ``(B, n_max, ne-1)``。

    每行剩余列为 ``cls`` + 归一化 ``xywh``；不足 ``n_max`` 的位置为 0，与
    ``v8DetectionLoss.preprocess`` 前半段一致。

    Parameters
    ----------
    targets
        ``(N, ne)``，第 0 列为样本所属 batch 内图像下标 ``0..B-1``，后面列为该条 GT 属性
        （通常为 ``cls`` + 4 维归一化 ``xywh``，即 ``ne=6``）。
    batch_size
        当前 batch 图像数 ``B``（可与 ``targets[:,0].max()+1`` 一致，由调用方传入）。
    device
        输出张量所在 device。

    Returns
    -------
    torch.Tensor
        ``(B, n_max, ne-1)``。第 ``j`` 行前若干列为第 ``j`` 张图的所有 GT（顺序与
        ``targets`` 中该行出现的顺序一致），其余槽位为 0；``n_max`` 为本 batch 中单张图
        GT 数的最大值。
    """
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
    return out


def build_v8_target_rows_from_flat_components(
    batch_idx: torch.Tensor,
    cls: torch.Tensor,
    bboxes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    将展平 GT 拼成与 Ultralytics 一致的行表 ``(N, 6)``。

    每行 ``[image_idx, cls, cx, cy, w, h]``，其中 ``cx,cy,w,h`` 为 **归一化 xywh**。
    无 GT 时返回形状 ``(0, 6)`` 的空表，仍供 ``group_flat_v8_rows_per_image_by_batch_id`` 走统一路径。
    """
    if batch_idx.numel() == 0:
        return torch.zeros((0, 6), device=device)
    return torch.cat(
        (
            batch_idx.to(device=device).view(-1, 1).long(),
            cls.to(device=device).view(-1, 1).long(),
            bboxes.to(device=device).view(-1, 4).float(),
        ),
        1,
    )


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


class DetectionLossYOLOv8:
    """
    见模块顶部的流程说明。对外入口：``forward_loss_vec`` → ``(3,)`` 的 ``[box, cls, dfl]``；
    ``__call__`` 返回三项之和。

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
        """懒初始化 TAL、``_BboxLoss``、DFL 投影向量；device 变化时重建。"""
        if self._init_device != device or self._assigner is None:
            self._assigner = TaskAlignedAssigner(
                topk=self._tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
            )
            self._bbox_loss = _BboxLoss(self.reg_max).to(device)
            self._proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
            self._init_device = device

    def _stride_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``self.stride`` 与当前 ``device/dtype`` 对齐。"""
        return self.stride.to(device=device, dtype=dtype)

    @staticmethod
    def _feature_list(preds: Union[List[torch.Tensor], Tuple[Any, ...]]) -> List[torch.Tensor]:
        """
        Detect 多尺度特征列表。

        - ``DAGNet`` 单输出节点时返回 ``(detect_out,)``，训练态 ``detect_out`` 即为特征 list。
        - 部分封装为 ``(aux, feats)`` 时取 ``feats``（``preds[1]``）。
        - ``Detect`` 在 eval 下可能返回 ``(decoded, raw_feats)``，损失需要 ``raw_feats``。
        """
        if not isinstance(preds, tuple):
            return preds
        if len(preds) == 1:
            inner = preds[0]
            if isinstance(inner, tuple) and len(inner) == 2:
                return inner[1]
            return inner
        return preds[1]

    @staticmethod
    def merge_multiscale_detect_features_to_flat_anchor_tensor(feats: List[torch.Tensor], no: int) -> torch.Tensor:
        """
        将多层 Detect 输出沿「全锚点」维拼成一张扁平大表。

        各尺度特征先按空间维展平，再在锚点维 ``cat``，等价于把 P3/P4/P5… 上所有网格位置
        排成一条长序列；结果不是二维 H×W 图，而是 ``(B, no, A)``，``A`` 为总锚点数。

        Parameters
        ----------
        feats
            多尺度特征列表，每层 ``(B, no, H_i, W_i)``，``no = nc + 4*reg_max``。
        no
            单层的通道数 ``no``（与 ``self.no`` 一致）。

        Returns
        -------
        merged_head_output
            形状 ``(B, no, A)``，其中 ``A = sum_i(H_i * W_i)`` 为所有层锚点总数。
        """
        batch_size = feats[0].shape[0]
        flattened_per_level = [level_feat.view(batch_size, no, -1) for level_feat in feats]
        return torch.cat(flattened_per_level, dim=2)

    @staticmethod
    def split_flat_anchor_tensor_to_cls_and_reg(
        merged_head_output: torch.Tensor, reg_max: int, nc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将合并后的检测头输出按通道拆成分类与回归（DFL）两支，并转为 ``(B, A, C)`` 布局。

        Parameters
        ----------
        merged_head_output
            ``merge_multiscale_detect_features_to_flat_anchor_tensor`` 的输出，``(B, no, A)``。
        reg_max
            DFL 分布长度。
        nc
            类别数。

        Returns
        -------
        pred_scores
            分类分支 logits，``(B, A, nc)``。
        pred_distri
            回归分支 logits，``(B, A, 4*reg_max)``（l/t/r/b 每边 ``reg_max`` 个 bin）。
        """
        pred_reg_distri, pred_cls_scores = merged_head_output.split(
            (reg_max * 4, nc), dim=1
        )
        pred_cls_scores = pred_cls_scores.permute(0, 2, 1).contiguous()
        pred_reg_distri = pred_reg_distri.permute(0, 2, 1).contiguous()
        return pred_cls_scores, pred_reg_distri

    def _decode_pred_boxes(
        self, anchor_points: torch.Tensor, pred_distri: torch.Tensor
    ) -> torch.Tensor:
        """
        回归分支 → 各锚点上的预测框（**网格坐标系** xyxy，尚未乘 stride）。

        - **use_dfl**：``(B, A, 4*reg_max)`` logits 经 softmax×投影得到期望 ltrb，形状变为 ``(B, A, 4)``。
        - **非 DFL**（``reg_max==1``）：``pred_distri`` 已可视作 ``(B, A, 4)`` 的 ltrb，不再做 softmax。

        Parameters
        ----------
        anchor_points
            ``(A, 2)``，全图锚点中心在网格坐标下的 ``(x, y)``。
        pred_distri
            ``(B, A, 4*reg_max)`` 或 ``(B, A, 4)``（见上）。

        Returns
        -------
        torch.Tensor
            ``(B, A, 4)``，与 ``anchor_points`` 同坐标系的 **xyxy**；后续 TAL 中会 ``* stride_tensor`` 到像素。
        """
        if self.use_dfl:
            b, a, c = pred_distri.shape
            pred_distri = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(self._proj.type(pred_distri.dtype))
        return _dist2bbox(pred_distri, anchor_points, xywh=False)

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

    def forward_loss_vec(
        self,
        preds: List[torch.Tensor],
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回 ``(3,)``：**[box, cls, dfl]**，已乘 ``hyp.*`` 与 ``batch_size``（与 v8 一致）。

        形状：``B`` batch，``no=nc+4*reg_max``，``A=Σ_i H_i W_i``，``N`` GT 条数，``n_max`` 单图 GT 上限。
        """
        # feats: List[(B, no, H_i, W_i)]
        feats = self._feature_list(preds)
        device = feats[0].device
        dtype = feats[0].dtype
        self._ensure_heads(device, dtype)
        # stride: (nl,)
        stride = self._stride_on(device, dtype)

        # merged_head: (B, no, A)
        merged_head = self.merge_multiscale_detect_features_to_flat_anchor_tensor(feats, self.no)
        # pred_scores: (B, A, nc), pred_distri: (B, A, 4*reg_max)
        pred_scores, pred_distri = self.split_flat_anchor_tensor_to_cls_and_reg(
            merged_head, self.reg_max, self.nc
        )
        batch_size = pred_scores.shape[0]
        # imgsz: (2,) = [H, W]，像素尺度
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]
        # anchor_points: (A, 2), stride_tensor: (A, 1)
        anchor_points, stride_tensor = build_flat_anchor_points_and_strides_from_multiscale_feats(feats, stride, 0.5)

        # raw_gt: (N, 6) = [img_idx, cls, cx, cy, w, h]（xywh 为归一化）
        raw_gt = build_v8_target_rows_from_flat_components(batch_idx, cls, bboxes, device)
        # scale_tensor: (4,) = [W, H, W, H]
        scale_tensor = imgsz[[1, 0, 1, 0]]
        # packed_gt: (B, n_max, 5) = [cls, cx, cy, w, h]（按图聚合并 padding）
        packed_gt = group_flat_v8_rows_per_image_by_batch_id(raw_gt, batch_size, device)
        # packed_gt[...,1:5]: (B, n_max, 4), 归一化 xywh -> 像素 xyxy
        packed_gt[..., 1:5] = _xywh2xyxy(packed_gt[..., 1:5].mul_(scale_tensor))
        # gt_labels: (B, n_max, 1), gt_bboxes: (B, n_max, 4) (pixel xyxy)
        gt_labels, gt_bboxes = packed_gt.split((1, 4), 2)
        # mask_gt: (B, n_max, 1), True 表示该槽位有 GT（非 padding）
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        # pred_bboxes: (B, A, 4), 网格坐标系 xyxy（尚未乘 stride）
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
        # tal.*（TAL 分配结果）
        # - tal.target_scores: (B, A, nc) 软分类标签（多数为背景 0）
        # - tal.target_bboxes_px: (B, A, 4) 像素空间目标框（与 gt_bboxes 同一坐标系）
        # - tal.fg_mask: (B, A, 1) 前景掩码（正样本为 True/1）
        target_scores_sum = max(tal.target_scores.sum(), 1)
        # 归一化分母：避免无前景时出现除零；是标量（0-d tensor）或 Python int。

        l_cls = self._loss_cls(pred_scores, tal.target_scores, target_scores_sum, dtype)
        # l_cls: 分类损失标量（未乘 hyp，且内部已按 target_scores_sum 归一化）
        l_box, l_dfl = self._loss_box_dfl(
            pred_distri,
            pred_bboxes,
            anchor_points,
            tal,
            target_scores_sum,
            stride_tensor,
        )
        # l_box: 回归（CIoU）标量；主要在 fg_mask 为 True 的锚点上计算
        # l_dfl: DFL（分布焦点）标量；reg_max==1 时通常为 0（见 _loss_box_dfl）

        loss = torch.zeros(3, device=device)
        # loss: (3,) 向量，对应 [box, cls, dfl]，最后会再乘 batch_size
        loss[0] = l_box * self.hyp.box
        loss[1] = l_cls * self.hyp.cls
        loss[2] = l_dfl * self.hyp.dfl
        # 返回 (3,)（含 hyp.* 权重），再乘 batch_size 以对齐 Ultralytics 的标度约定
        return loss * batch_size

    def __call__(
        self,
        preds: List[torch.Tensor],
        *,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """标量总损失，等价于 ``forward_loss_vec(...).sum()``。"""
        if not isinstance(preds, list) and not isinstance(preds, tuple):
            raise ValueError("preds 须为 Detect 训练输出的特征层列表（或含 preds 的 tuple）。")
        return self.forward_loss_vec(preds, batch_idx=batch_idx, cls=cls, bboxes=bboxes).sum()


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
