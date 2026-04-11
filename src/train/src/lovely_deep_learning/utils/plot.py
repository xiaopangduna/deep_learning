"""图像拼接、可视化辅助（BGR / OpenCV 约定）。"""

from __future__ import annotations

import cv2
import numpy as np


def hstack_bgr_left_right_resize_right_to_match_left_with_gray_separator(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
) -> np.ndarray:
    """
    水平拼接 ``left_bgr | 灰条 | right_bgr``（BGR，``H×W×3``）。

    若左右 ``(H,W)`` 不一致，将 **右侧** 双线性缩放到与 **左侧** 相同的高与宽；中间为固定宽度灰度分隔列。
    """
    h, wl = left_bgr.shape[:2]
    hr, wr = right_bgr.shape[:2]
    if (h, wl) != (hr, wr):
        right_bgr = cv2.resize(right_bgr, (wl, h), interpolation=cv2.INTER_LINEAR)
    gap_w = 3
    sep = np.full((h, gap_w, 3), 220, dtype=np.uint8)
    return np.hstack([left_bgr, sep, right_bgr])
