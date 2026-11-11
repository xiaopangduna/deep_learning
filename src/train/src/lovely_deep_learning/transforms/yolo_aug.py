"""YOLOv8 训练增强的 numpy 路径。

Mosaic 画布是 ``2s×2s``。官方 ``RandomPerspective`` 用负的 mosaic border，
一次 ``cv2.warpAffine`` 直接写出 ``s×s``，不再在 1280 上做 torchvision 仿射。
HSV 在 BGR uint8 上原地做，和 Ultralytics ``RandomHSV`` 同一套 LUT。
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .letterbox import letterbox_resize_and_pad


def hsv_bgr_(
    img: np.ndarray,
    hgain: float,
    sgain: float,
    vgain: float,
) -> np.ndarray:
    """原地调整 BGR uint8 的 H/S/V。增益为 0 时不改图像。"""
    if img.size == 0 or img.shape[-1] != 3:
        return img
    if not (hgain or sgain or vgain):
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
    lut_sat[0] = 0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return img


def maybe_fliplr(
    img: np.ndarray,
    boxes: np.ndarray,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """以概率 ``p`` 左右翻转图像和像素 XYXY。"""
    if p <= 0.0 or random.random() >= p:
        return img, boxes
    out = np.ascontiguousarray(np.fliplr(img))
    if boxes.size == 0:
        return out, boxes.reshape(0, 4)
    w = float(out.shape[1])
    flipped = boxes.copy()
    x1 = flipped[:, 0].copy()
    x2 = flipped[:, 2].copy()
    flipped[:, 0] = w - x2
    flipped[:, 2] = w - x1
    return out, flipped


def letterbox_bgr(
    img: np.ndarray,
    boxes: np.ndarray,
    size: int,
    fill: int = 114,
) -> tuple[np.ndarray, np.ndarray]:
    """保比例缩放到不超过 ``size``，再 pad 成正方形。框跟着缩放和平移。"""
    h, w = int(img.shape[0]), int(img.shape[1])
    resize_h, resize_w, left, top, right, bottom = letterbox_resize_and_pad(
        h, w, int(size), int(size), scaleup=True, center=True
    )
    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = np.asarray(boxes, dtype=np.float32).copy()
    if (resize_h, resize_w) != (h, w):
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        if boxes.size:
            boxes[:, [0, 2]] *= resize_w / float(w)
            boxes[:, [1, 3]] *= resize_h / float(h)
    if left or top or right or bottom:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(fill, fill, fill)
        )
        if boxes.size:
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
    return img, boxes


def _box_candidates(
    box1: np.ndarray,
    box2: np.ndarray,
    wh_thr: float = 2,
    ar_thr: float = 100,
    area_thr: float = 0.1,
    eps: float = 1e-16,
) -> np.ndarray:
    """与 Ultralytics ``RandomPerspective.box_candidates`` 相同。``box*`` 为 ``(4, N)``。"""
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def _apply_bboxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """仿射矩阵作用到 XYXY 的四个角，再取外接矩形。"""
    n = len(boxes)
    if n == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    xy = np.ones((n * 4, 3), dtype=np.float32)
    xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ matrix.T
    xy = xy[:, :2].reshape(n, 8)
    xs = xy[:, [0, 2, 4, 6]]
    ys = xy[:, [1, 3, 5, 7]]
    return np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), axis=1).astype(np.float32)


def random_affine(
    img: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    *,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    border: tuple[int, int] = (0, 0),
    fill: int = 114,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """官方 ``RandomPerspective`` 在 shear=0、perspective=0 时的仿射。

    ``border`` 是 ``(上下, 左右)``。Mosaic 传 ``(-s/2, -s/2)``，输出边长就是 ``s``。
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    height, width = int(img.shape[0]), int(img.shape[1])
    out_w = int(width + border[1] * 2)
    out_h = int(height + border[0] * 2)

    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -width / 2
    center[1, 2] = -height / 2

    rotation = np.eye(3, dtype=np.float32)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    rotation[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    translation = np.eye(3, dtype=np.float32)
    translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_w
    translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_h

    matrix = translation @ rotation @ center
    changed = (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any()
    if changed:
        out = cv2.warpAffine(
            img, matrix[:2], dsize=(out_w, out_h), borderValue=(fill, fill, fill)
        )
    else:
        out = img.copy()

    new_boxes = _apply_bboxes(boxes, matrix)
    if new_boxes.size:
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, out_w)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, out_h)
    scaled = boxes.copy()
    if scaled.size:
        scaled *= scale_factor
    if new_boxes.size == 0:
        keep = np.zeros((0,), dtype=bool)
    else:
        keep = _box_candidates(scaled.T, new_boxes.T)
    return out, new_boxes[keep], classes[keep]
