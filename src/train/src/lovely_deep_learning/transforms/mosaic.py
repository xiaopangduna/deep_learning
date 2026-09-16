"""YOLOv8 风格 4 图 Mosaic：长边缩到 ``s`` 后贴到 ``2s×2s`` 画布。

中心点按 ``xc, yc ~ Uniform(-s/2, 2s + s/2)``（即 ``mosaic_border = s/2`` 时的
``uniform(-border, 2s+border)``）。画布填 114；框平移后 clip，丢掉宽或高 ``≤ 1`` 的框。

输入图像为 RGB ``uint8`` ``(H, W, C)``；框为像素 XYXY。
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import cv2
import numpy as np


def sample_mosaic_center(imgsz: int, rng: Any = None) -> tuple[int, int]:
    """``xc, yc ~ Uniform(-s/2, 2s + s/2)``。"""
    s = int(imgsz)
    half = s / 2.0
    low, high = -half, 2.0 * s + half
    if rng is None:
        xc = int(random.uniform(low, high))
        yc = int(random.uniform(low, high))
    else:
        xc = int(rng.uniform(low, high))
        yc = int(rng.uniform(low, high))
    return xc, yc


def resize_long_side(
    img: np.ndarray,
    bboxes: np.ndarray,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """长边缩到 ``imgsz``，与官方 ``load_image(rect_mode=True)`` 取整一致。"""
    h0, w0 = int(img.shape[0]), int(img.shape[1])
    r = float(imgsz) / float(max(h0, w0))
    if r == 1.0:
        return img, bboxes
    new_w = min(math.ceil(w0 * r), int(imgsz))
    new_h = min(math.ceil(h0 * r), int(imgsz))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if bboxes.size == 0:
        return out, bboxes.reshape(0, 4).astype(np.float32)
    boxes = np.asarray(bboxes, dtype=np.float32).copy()
    sx = new_w / float(w0)
    sy = new_h / float(h0)
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return out, boxes


def mosaic4(
    images: Sequence[np.ndarray],
    bboxes: Sequence[np.ndarray],
    classes: Sequence[np.ndarray],
    imgsz: int = 640,
    fill: int = 114,
    rng: Any = None,
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拼 4 张图到 ``2s×2s``。返回 RGB 画布、XYXY 框、类别。"""
    if len(images) != 4 or len(bboxes) != 4 or len(classes) != 4:
        raise ValueError("mosaic4 需要恰好 4 张图及其框、类别")
    s = int(imgsz)
    canvas = 2 * s
    xc, yc = center if center is not None else sample_mosaic_center(s, rng)
    img4 = None
    all_boxes: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []

    for i in range(4):
        img, boxes = resize_long_side(
            images[i],
            np.asarray(bboxes[i], dtype=np.float32).reshape(-1, 4),
            s,
        )
        cls_i = np.asarray(classes[i]).reshape(-1)
        h, w = img.shape[0], img.shape[1]
        if img4 is None:
            ch = img.shape[2] if img.ndim == 3 else 1
            img4 = np.full((canvas, canvas, ch), fill, dtype=np.uint8)

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, canvas), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(canvas, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, canvas), min(canvas, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 中心可在画布外：负索引会从末尾回绕，须 clip 到 [0, canvas] 再贴。
        x1a_d = int(max(0, x1a))
        y1a_d = int(max(0, y1a))
        x2a_d = int(min(canvas, x2a))
        y2a_d = int(min(canvas, y2a))
        dw, dh = x2a_d - x1a_d, y2a_d - y1a_d
        if dw > 0 and dh > 0:
            x1b_d = int(x1b + (x1a_d - x1a))
            y1b_d = int(y1b + (y1a_d - y1a))
            x2b_d = x1b_d + dw
            y2b_d = y1b_d + dh
            ih, iw = img.shape[0], img.shape[1]
            x1b_d = max(0, x1b_d)
            y1b_d = max(0, y1b_d)
            x2b_d = min(iw, x2b_d)
            y2b_d = min(ih, y2b_d)
            dw = min(dw, x2b_d - x1b_d)
            dh = min(dh, y2b_d - y1b_d)
            if dw > 0 and dh > 0:
                img4[y1a_d : y1a_d + dh, x1a_d : x1a_d + dw] = img[
                    y1b_d : y1b_d + dh, x1b_d : x1b_d + dw
                ]
        padw = x1a - x1b
        padh = y1a - y1b
        if boxes.size:
            shifted = boxes.copy()
            shifted[:, [0, 2]] += padw
            shifted[:, [1, 3]] += padh
            all_boxes.append(shifted)
            all_cls.append(cls_i)

    if not all_boxes:
        empty_b = np.zeros((0, 4), dtype=np.float32)
        empty_c = np.zeros((0,), dtype=np.int32)
        return img4, empty_b, empty_c

    boxes_cat = np.concatenate(all_boxes, axis=0).astype(np.float32)
    cls_cat = np.concatenate(all_cls, axis=0)
    boxes_cat[:, [0, 2]] = boxes_cat[:, [0, 2]].clip(0, canvas)
    boxes_cat[:, [1, 3]] = boxes_cat[:, [1, 3]].clip(0, canvas)
    keep = (boxes_cat[:, 2] - boxes_cat[:, 0] > 1.0) & (
        boxes_cat[:, 3] - boxes_cat[:, 1] > 1.0
    )
    return img4, boxes_cat[keep], np.asarray(cls_cat[keep], dtype=np.int32)
