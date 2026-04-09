# -*- encoding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Callable, List

import numpy as np
import pandas as pd
import torch
import cv2
from torchvision import tv_tensors
from torchvision.ops import batched_nms

from .base import BaseDataset
from ..nn.head import Detect


def iou_xyxy_pair(a: np.ndarray, b: np.ndarray) -> float:
    """单对 XYXY 框的 IoU，``a``/``b`` 为 ``(4,)``。"""
    ax1, ay1, ax2, ay2 = (float(a[i]) for i in range(4))
    bx1, by1, bx2, by2 = (float(b[i]) for i in range(4))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-7
    return inter / union


def greedy_match_pred_gt_iou(
    bboxes_pred: np.ndarray,
    classes_pred: np.ndarray,
    bboxes_gt: np.ndarray,
    classes_gt: np.ndarray,
    iou_threshold: float,
    pred_order: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    同类内按 IoU 贪心匹配：每个 GT 最多匹配一个预测，每个预测最多匹配一个 GT。
    返回 ``pred_matched``, ``gt_matched``，长度分别为 Np、Ng 的 bool 数组。
    """
    n_p = int(bboxes_pred.shape[0])
    n_g = int(bboxes_gt.shape[0])
    pred_matched = np.zeros(n_p, dtype=bool)
    gt_matched = np.zeros(n_g, dtype=bool)
    gt_taken = np.zeros(n_g, dtype=bool)
    if pred_order is None:
        pred_order = np.arange(n_p)
    for pi in pred_order:
        best_j, best_iou = -1, -1.0
        cp = int(classes_pred[pi])
        for gj in range(n_g):
            if gt_taken[gj]:
                continue
            if cp != int(classes_gt[gj]):
                continue
            iou = iou_xyxy_pair(bboxes_pred[pi], bboxes_gt[gj])
            if iou > best_iou:
                best_iou = iou
                best_j = gj
        if best_j >= 0 and best_iou >= iou_threshold:
            pred_matched[pi] = True
            gt_taken[best_j] = True
    gt_matched[:] = gt_taken
    return pred_matched, gt_matched


def cxcywh_pixels_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """``boxes``: ``(..., 4)`` 中心 cxcywh 像素 → xyxy。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
    )


def xyxy_abs_pixels_to_xywh_norm(
    xyxy: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """``xyxy`` 像素框转归一化 ``xywh``（中心点 + 宽高）。"""
    x1 = xyxy[..., 0]
    y1 = xyxy[..., 1]
    x2 = xyxy[..., 2]
    y2 = xyxy[..., 3]
    w = float(width)
    h = float(height)
    cx = (x1 + x2) * 0.5 / w
    cy = (y1 + y2) * 0.5 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return torch.stack((cx, cy, bw, bh), dim=-1)


def apply_nms_to_detections(
    dets: torch.Tensor, conf_thres: float, nms_iou: float
) -> torch.Tensor:
    """
    对 Detect 后处理输出做按类 ``batched_nms``，返回固定形状 ``(B, M, 6)``（不足行补零）。

    维度与语义约定：
    - 输入 ``dets``: ``(B, M, 6)``
      - ``B``: batch size
      - ``M``: 每张图解码后的候选框数上限（max_det）
      - ``6``: ``[cx, cy, w, h, score, cls_id]``（坐标单位为像素）
    - 输出 ``out``: ``(B, M, 6)``
      - 对每张图，前 ``n`` 行为 NMS 保留结果，剩余 ``M-n`` 行为 0 填充
    """
    B, M, _ = dets.shape
    device, dtype = dets.device, dets.dtype
    out = torch.zeros(B, M, 6, device=device, dtype=dtype)
    max_keep = M
    for b in range(B):
        row = dets[b]  # (M, 6)
        # 置信度阈值过滤后：从 (M, 6) -> (K, 6)，K<=M
        mask = row[:, 4] > conf_thres
        if not mask.any():
            continue
        xywh = row[mask, :4]          # (K, 4), cxcywh(pixel)
        scores = row[mask, 4]         # (K,)
        cls_ids = row[mask, 5].long()  # (K,)
        boxes_xyxy = cxcywh_pixels_to_xyxy(xywh)  # (K, 4), xyxy(pixel)
        keep = batched_nms(boxes_xyxy, scores, cls_ids, nms_iou)
        keep = keep[:max_keep]  # (N,), N<=K<=M
        if keep.numel() == 0:
            continue
        xywh_k = xywh[keep]              # (N, 4)
        sc_k = scores[keep]              # (N,)
        cl_k = cls_ids[keep].to(dtype=dtype)  # (N,)
        n = keep.numel()
        # 写回固定形状输出：out[b] 仍为 (M, 6)，仅前 n 行有效
        out[b, :n, :4] = xywh_k
        out[b, :n, 4] = sc_k
        out[b, :n, 5] = cl_k
    return out


def postprocess_detections(
    raw: torch.Tensor,
    max_det: int,
    nc: int,
    nms: bool,
    conf_thres: float,
    nms_iou: float,
) -> torch.Tensor:
    """Detect 解码 + 可选按类 NMS，返回 ``(B, max_det, 6)``。"""
    dets = Detect.postprocess(raw.permute(0, 2, 1), max_det, nc)
    if nms:
        dets = apply_nms_to_detections(dets, conf_thres=conf_thres, nms_iou=nms_iou)
    return dets


class ObjectDetectDataset(BaseDataset):

    def __init__(
        self,
        csv_paths: Sequence[Union[str, Path]],
        key_map: Dict[str, str] = {"img_path": "path_img",
                                   "object_label_path": "path_label_detect_yolo"},
        transform: Optional[Callable] = None,
        map_class_id_to_class_name: Optional[Union[Dict[Any, str], str]] = None,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
        nms: bool = True,
        nms_iou: float = 0.7,
        inference_conf_thres: float = 0.001,
    ):
        super().__init__(csv_paths=csv_paths, key_map=key_map, transform=transform)
        self.map_class_id_to_class_name = map_class_id_to_class_name
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.nms = bool(nms)
        self.nms_iou = float(nms_iou)
        self.inference_conf_thres = float(inference_conf_thres)
        self._has_label = "object_label_path" in self.sample_path_table.columns

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        img_path = str(self.sample_path_table["img_path"].iloc[index])
        img_np, img_shape = BaseDataset.read_img(img_path, None)
        img_tensor = BaseDataset.convert_img_from_numpy_to_tensor_uint8(img_np)
        img_tv = tv_tensors.Image(img_tensor)
        net_in: Dict[str, Any] = {
            "img_path": img_path,
            "img_shape": img_shape,
        }

        if not self._has_label:
            net_in["img_tv_transformed"] = (
                self.transform(img_tv) if self.transform else img_tv
            )
            return net_in, {}

        object_label_path = str(
            self.sample_path_table["object_label_path"].iloc[index])
        cls, bboxes_cxcywh_rel = ObjectDetectDataset.read_yolo_detection_labels(
            object_label_path
        )
        bboxes_abs_xyxy = (
            ObjectDetectDataset.convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
                bboxes_cxcywh_rel, img_shape
            )
        )
        cls_tensor = torch.from_numpy(cls)
        bboxes_tensor = torch.from_numpy(
            bboxes_abs_xyxy.astype(np.float32, copy=False)
        )
        bboxes_tv = tv_tensors.BoundingBoxes(
            bboxes_tensor,
            format="XYXY",
            canvas_size=(img_shape[0], img_shape[1]),
        )
        target = {"cls": cls_tensor, "bboxes": bboxes_tv}
        if self.transform:
            img_tv_transformed, t = self.transform(img_tv, target)
            cls_tensor_transformed = t["cls"]
            bboxes_tv_transformed = t["bboxes"]
        else:
            img_tv_transformed = img_tv
            cls_tensor_transformed = cls_tensor
            bboxes_tv_transformed = bboxes_tv

        net_in["img_tv_transformed"] = img_tv_transformed
        net_out = {
            "cls_np": cls,
            "bboxes_cxcywh_rel_np": bboxes_cxcywh_rel,
            "cls_tv_transformed": cls_tensor_transformed,
            "bboxes_xyxy_abs_tv_transformed": bboxes_tv_transformed,
        }
        return net_in, net_out

    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(x):
            net_in_tuple, net_out_tuple = list(zip(*x))
            net_in_list: list[Dict[str, Any]] = []
            net_out_list: list[Dict[str, Any]] = []
            for i, (ni0, no0) in enumerate(zip(net_in_tuple, net_out_tuple)):
                ni = dict(ni0)
                img_t = (
                    ni["img_tv_transformed"].data
                    if hasattr(ni["img_tv_transformed"], "data")
                    else ni["img_tv_transformed"]
                )
                ni["img"] = img_t
                if not no0:
                    net_in_list.append(ni)
                    net_out_list.append({})
                    continue
                no = dict(no0)
                _c, h_i, w_i = img_t.shape
                cls_t = no["cls_tv_transformed"]
                boxes = no["bboxes_xyxy_abs_tv_transformed"]
                if hasattr(boxes, "data"):
                    boxes = boxes.data
                elif hasattr(boxes, "as_tensor"):
                    boxes = boxes.as_tensor()
                n = int(cls_t.shape[0])
                if n == 0:
                    net_in_list.append(ni)
                    net_out_list.append(no)
                    continue
                xywh = xyxy_abs_pixels_to_xywh_norm(
                    boxes.float(), int(h_i), int(w_i)
                )
                device = xywh.device
                no["batch_idx"] = torch.full(
                    (n,), float(i), device=device, dtype=torch.float32
                )
                no["bboxes_xywh_norm"] = xywh
                net_in_list.append(ni)
                net_out_list.append(no)
            return (tuple(net_in_list), tuple(net_out_list))

        return collate_fn

    @staticmethod
    def draw_label_on_numpy(
        image: np.ndarray,
        bboxes: np.ndarray,
        classes: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = [(0, 255, 0)],
        thickness: int = 2,
        scores: Optional[np.ndarray] = None,
        box_colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        在图像上绘制边界框（**仅支持像素 XYXY**）。

        参数:
            image: BGR 图像，``(H, W, C)``
            bboxes: ``(N, 4)``，每行 ``x1, y1, x2, y2`` 像素坐标，与 ``image`` 尺寸一致
            classes: 类别 ID，长度 ``N``
            class_names: 可选 id→名称
            colors: BGR 颜色列表
            thickness: 线宽
            scores: 可选，与框一一对应的置信度，会拼在标签文字后
            box_colors: 可选，``(N, 3)`` 逐框 BGR；提供时优先于 ``colors`` 按类着色逻辑

        返回:
            绘制后的图像（就地修改 ``image`` 并返回同一数组）
        """
        h_img, w_img = image.shape[:2]

        if colors is None:
            colors = [(0, 255, 0)]

        for i, (bbox, cls_id) in enumerate(zip(bboxes, classes)):
            x1, y1, x2, y2 = (float(bbox[j]) for j in range(4))
            x1 = int(round(max(0, min(x1, w_img - 1))))
            x2 = int(round(max(0, min(x2, w_img - 1))))
            y1 = int(round(max(0, min(y1, h_img - 1))))
            y2 = int(round(max(0, min(y2, h_img - 1))))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # 获取当前框的颜色
            if box_colors is not None and i < box_colors.shape[0]:
                color = (
                    int(box_colors[i, 0]),
                    int(box_colors[i, 1]),
                    int(box_colors[i, 2]),
                )
            elif len(colors) > 1:
                color = colors[cls_id % len(colors)]
            else:
                color = colors[0]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # 添加类别标签
            label = f"Class: {cls_id}"
            if class_names and cls_id in class_names:
                label = class_names[cls_id]
            if scores is not None and i < len(scores):
                label = f"{label} {float(scores[i]):.2f}"

            # 计算标签背景框的大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

            # 绘制标签背景
            cv2.rectangle(
                image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

            # 根据标签背景亮度自动选择文字颜色，提升可读性
            luminance = 0.114 * float(color[0]) + 0.587 * float(color[1]) + 0.299 * float(color[2])
            text_color = (20, 20, 20) if luminance >= 150 else (245, 245, 245)

            # 绘制标签文字
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA
            )

        return image

    @staticmethod
    def draw_target_and_predict_label_on_numpy(
        image: np.ndarray,
        bboxes_pred: np.ndarray,
        classes_pred: np.ndarray,
        bboxes_gt: np.ndarray,
        classes_gt: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        pred_scores: Optional[np.ndarray] = None,
        match_by_iou: bool = True,
        iou_match_threshold: float = 0.5,
        color_match: Tuple[int, int, int] = (80, 240, 120),
        color_mismatch: Tuple[int, int, int] = (30, 30, 210),
        color_pred: Tuple[int, int, int] = (0, 140, 255),
        color_gt: Tuple[int, int, int] = (80, 240, 120),
        gap_px: int = 3,
        gap_color: Tuple[int, int, int] = (0, 0, 0),
        thickness: int = 2,
        cached_match: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        将同一张底图复制两份左右拼接：**左侧为预测框**，**右侧为真值框**。

        当 ``match_by_iou=True``（默认）时，按 **类别相同且 IoU ≥ 阈值** 做贪心一一匹配：
        匹配上的预测框与真值框用 ``color_match``（默认绿），否则用 ``color_mismatch``（默认红）。
        预测框处理顺序：若提供 ``pred_scores``，按置信度从高到低优先匹配。

        当 ``match_by_iou=False`` 时，左侧统一 ``color_pred``、右侧统一 ``color_gt``（旧行为）。

        参数:
            image: BGR ``(H, W, 3)``，不在此函数内被原地修改；返回新图。
            bboxes_pred / classes_pred: 预测框与类别，像素 XYXY。
            bboxes_gt / classes_gt: 真值框与类别。
            class_names: 可选类别 id→名称。
            pred_scores: 可选，与预测框等长的置信度。
            match_by_iou: 是否按 IoU+类别着色。
            iou_match_threshold: IoU 阈值。
            color_match / color_mismatch: 判定一致 / 不一致时的 BGR 颜色。
            color_pred / color_gt: 仅在 ``match_by_iou=False`` 时使用。
            gap_px: 左右图之间的分隔条宽度。
            gap_color: 分隔条颜色（BGR）。
            thickness: 线宽。
            cached_match: 可选 ``(pred_matched, gt_matched)`` 与 :func:`greedy_match_pred_gt_iou`
                返回值同形；传入时不再重复计算匹配（便于与外部逻辑共用同一结果）。

        返回:
            拼接后的 BGR 图像 ``(H, 2*W+gap_px, 3)``（左右同高；若宽高不一致则将右侧缩放到与左侧一致）。
        """
        left = image.copy()
        right = image.copy()
        bboxes_pred = np.asarray(bboxes_pred, dtype=np.float32).reshape(-1, 4)
        classes_pred = np.asarray(classes_pred).reshape(-1)
        bboxes_gt = np.asarray(bboxes_gt, dtype=np.float32).reshape(-1, 4)
        classes_gt = np.asarray(classes_gt).reshape(-1)
        n_p, n_g = bboxes_pred.shape[0], bboxes_gt.shape[0]

        if match_by_iou:
            if cached_match is not None:
                pred_ok, gt_ok = cached_match
            else:
                pred_order = None
                if pred_scores is not None and n_p > 0:
                    pred_order = np.argsort(-np.asarray(pred_scores, dtype=np.float32))
                pred_ok, gt_ok = greedy_match_pred_gt_iou(
                    bboxes_pred,
                    classes_pred,
                    bboxes_gt,
                    classes_gt,
                    float(iou_match_threshold),
                    pred_order=pred_order,
                )
            cm = np.array(color_match, dtype=np.uint8)
            cx = np.array(color_mismatch, dtype=np.uint8)
            pred_colors = np.tile(cx, (max(n_p, 1), 1))[:n_p] if n_p else np.zeros((0, 3), dtype=np.uint8)
            gt_colors = np.tile(cx, (max(n_g, 1), 1))[:n_g] if n_g else np.zeros((0, 3), dtype=np.uint8)
            if n_p:
                pred_colors[pred_ok] = cm
            if n_g:
                gt_colors[gt_ok] = cm
            ObjectDetectDataset.draw_label_on_numpy(
                left,
                bboxes_pred,
                classes_pred,
                class_names=class_names,
                colors=[color_pred],
                thickness=thickness,
                scores=pred_scores,
                box_colors=pred_colors,
            )
            ObjectDetectDataset.draw_label_on_numpy(
                right,
                bboxes_gt,
                classes_gt,
                class_names=class_names,
                colors=[color_gt],
                thickness=thickness,
                scores=None,
                box_colors=gt_colors,
            )
        else:
            ObjectDetectDataset.draw_label_on_numpy(
                left,
                bboxes_pred,
                classes_pred,
                class_names=class_names,
                colors=[color_pred],
                thickness=thickness,
                scores=pred_scores,
            )
            ObjectDetectDataset.draw_label_on_numpy(
                right,
                bboxes_gt,
                classes_gt,
                class_names=class_names,
                colors=[color_gt],
                thickness=thickness,
                scores=None,
            )
        h, wl = left.shape[:2]
        hr, wr = right.shape[:2]
        if (h, wl) != (hr, wr):
            right = cv2.resize(right, (wl, h), interpolation=cv2.INTER_LINEAR)
        sep = np.full((h, max(gap_px, 0), 3), np.asarray(gap_color, dtype=np.uint8), dtype=np.uint8)
        return np.hstack([left, sep, right])

    @staticmethod
    def read_yolo_detection_labels(file_path: str):
        """
        从YOLO格式的.txt文件中读取目标检测标签，并转换为NumPy数组。

        此函数假设标签文件格式正确，不包含任何校验。

        Args:
            file_path (str): YOLO标签文件的路径。

        Returns:
            cls: ``(N,)`` int32，类别 id。
            bbox: ``(N, 4)`` float32，YOLO txt 原始归一化 **CXCYWH**（相对宽高）。
        """
        if not file_path:
            labels = np.empty((0, 5), dtype=np.float32)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                labels = [x.split()
                          for x in f.read().strip().splitlines() if len(x)]
                labels = np.array(labels, dtype=np.float32)

        cls = labels[:, 0].astype(np.int32)
        bbox = labels[:, 1:5].astype(np.float32)
        return cls, bbox

    @staticmethod
    def convert_cxcywh_relative_to_absolute(
        bboxes_rel: np.ndarray, img_shape: tuple
    ) -> np.ndarray:
        """
        归一化 CXCYWH → 像素 CXCYWH（相对宽高分别乘 ``W``、``H``）。
        """
        H, W = img_shape[0], img_shape[1]
        scale = np.array([W, H, W, H], dtype=np.float32)
        return np.asarray(bboxes_rel, dtype=np.float32) * scale

    @staticmethod
    def convert_cxcywh_to_xyxy_absolute(bboxes_abs_cxcywh: np.ndarray) -> np.ndarray:
        """
        像素 CXCYWH → 像素 XYXY。
        """
        t = np.asarray(bboxes_abs_cxcywh, dtype=np.float32)
        x_c, y_c, bw, bh = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        x1 = x_c - bw / 2
        y1 = y_c - bh / 2
        x2 = x_c + bw / 2
        y2 = y_c + bh / 2
        return np.column_stack((x1, y1, x2, y2))

    @staticmethod
    def convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
        bboxes_rel: np.ndarray, img_shape: tuple
    ) -> np.ndarray:
        """
        归一化 CXCYWH → 像素 XYXY（等价于先
        ``convert_cxcywh_relative_to_absolute`` 再 ``convert_cxcywh_absolute_to_xyxy``）。
        """
        abs_cxcywh = ObjectDetectDataset.convert_cxcywh_relative_to_absolute(
            bboxes_rel, img_shape
        )
        return ObjectDetectDataset.convert_cxcywh_to_xyxy_absolute(abs_cxcywh)

    def convert_img_from_tensor_to_numpy(self, img: torch.Tensor) -> np.ndarray:
        """
        将标准化的tensor转换为uint8的numpy数组，并进行反标准化处理

        Args:
            img: 输入的标准化tensor，格式为(C, H, W)

        Returns:
            反标准化后的numpy数组，格式为(H, W, C)，值域为[0, 255]
        """
        # 获取设备信息并复制tensor到CPU
        img = img.detach().cpu()

        # 获取tensor的值范围
        img_min = img.min().item()
        img_max = img.max().item()

        # 根据值的范围判断tensor类型并进行相应处理
        if img_min >= 0 and img_max <= 1.0:
            # 情况1: [0, 1] 范围的tensor
            img_clamped = torch.clamp(img, 0, 1)
        elif img_min >= 0 and img_max <= 255:
            # 情况2: [0, 255] 范围的tensor
            img_clamped = torch.clamp(img, 0, 255) / 255.0  # 归一化到[0,1]
        else:
            # 情况3: 标准化的tensor，需要反标准化
            if self.norm_mean is not None and self.norm_std is not None:
                # 反归一化: img * std + mean
                mean = torch.tensor(self.norm_mean).view(3, 1, 1)
                std = torch.tensor(self.norm_std).view(3, 1, 1)
                img = img * std + mean
            # 确保值在合理范围内
            img_clamped = torch.clamp(img, 0, 1)

        # 将[0,1]范围的值转换为uint8格式 (0~1 -> 0~255)
        img_uint8 = (img_clamped * 255).to(torch.uint8)

        # 调用基类的转换函数
        return BaseDataset.convert_img_from_tensor_to_numpy_uint8(img_uint8)
