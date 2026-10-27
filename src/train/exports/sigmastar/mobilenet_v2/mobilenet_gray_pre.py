# -*- coding: utf-8 -*-
"""Gray MobileNetV2 calibration preprocess.

Matches training val transform in
image_classifiter_from_detect_box_head_eye_open_mobilenet_v2_v001.yaml:
LetterBox 224 (fill 114, center, scaleup) -> Grayscale(1) -> /255 to [0, 1].
No mean/std. Output is NHWC (1, H, W, 1), same layout as eyeState_pre.py.
"""
import os

import cv2
import numpy as np
from PIL import Image


def _letterbox(rgb, dst_h, dst_w, fill=114):
    """Same pad geometry as lovely_deep_learning.transforms.letterbox."""
    src_h, src_w = rgb.shape[:2]
    r = min(dst_h / src_h, dst_w / src_w)
    resize_w = int(round(src_w * r))
    resize_h = int(round(src_h * r))
    dw = (dst_w - resize_w) / 2.0
    dh = (dst_h - resize_h) / 2.0
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    resized = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(fill, fill, fill),
    )


def _to_gray_u8(rgb):
    """torchvision Grayscale weights, truncated to uint8 like .to(uint8)."""
    gray = (
        0.2989 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    return np.clip(gray, 0, 255).astype(np.uint8)


def sim_preprocess(image_file, height=224, width=224, norm=True):
    with Image.open(image_file) as img:
        rgb = np.asarray(img.convert("RGB"))
    boxed = _letterbox(rgb, height, width, fill=114)
    gray = _to_gray_u8(boxed)
    im = np.expand_dims(gray, (0, 3))  # (1, H, W, 1)
    im = np.ascontiguousarray(im)
    if norm:
        return (im.astype(np.float32) / 255.0)
    return im


def image_preprocess(img_path, norm=True):
    height = int(os.environ.get("CLS_IN_HEIGHT", "224"))
    width = int(os.environ.get("CLS_IN_WIDTH", "224"))
    return sim_preprocess(img_path, height, width, norm=norm)
