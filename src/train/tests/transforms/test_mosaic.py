"""Mosaic4：中心采样区间、画布尺寸、框 clip。"""

from __future__ import annotations

import numpy as np

from lovely_deep_learning.transforms.mosaic import mosaic4, sample_mosaic_center


class _RecordRng:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[tuple[float, float]] = []

    def uniform(self, a, b):
        self.calls.append((float(a), float(b)))
        return self.value


def test_sample_mosaic_center_official_interval():
    s = 640
    rng = _RecordRng(0.0)
    sample_mosaic_center(s, rng)
    assert rng.calls[0] == (-s / 2, 2 * s + s / 2)
    assert rng.calls[1] == (-s / 2, 2 * s + s / 2)


def test_mosaic4_canvas_and_clipped_boxes():
    s = 64
    imgs = []
    boxes = []
    clses = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors):
        im = np.zeros((s, s, 3), dtype=np.uint8)
        im[:] = color
        imgs.append(im)
        boxes.append(np.array([[5.0, 5.0, 20.0, 20.0]], dtype=np.float32))
        clses.append(np.array([i], dtype=np.int32))
    out, xyxy, cls = mosaic4(
        imgs, boxes, clses, imgsz=s, center=(s + 16, s + 16)
    )
    assert out.shape == (2 * s, 2 * s, 3)
    assert int(out[0, 0, 0]) == 114
    assert int(out[s + 16 - 1, s + 16 - 1, 0]) == 255
    assert xyxy.ndim == 2 and xyxy.shape[1] == 4
    assert xyxy.shape[0] == cls.shape[0]
    assert (xyxy[:, 2] - xyxy[:, 0] > 1).all()
    assert (xyxy[:, 3] - xyxy[:, 1] > 1).all()
    assert (xyxy[:, 0] >= 0).all() and (xyxy[:, 1] >= 0).all()
    assert (xyxy[:, 2] <= 2 * s).all() and (xyxy[:, 3] <= 2 * s).all()


def test_mosaic4_drops_tiny_boxes_after_clip():
    s = 32
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    # 贴在画布外的框，clip 后宽高为 0
    boxes_tl = np.array([[-40.0, -40.0, -30.0, -30.0]], dtype=np.float32)
    out, xyxy, cls = mosaic4(
        [img, img, img, img],
        [boxes_tl, np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32)],
        [np.array([1], np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32), np.zeros((0,), np.int32)],
        imgsz=s,
        center=(s, s),
    )
    assert out.shape == (64, 64, 3)
    assert xyxy.shape[0] == 0
    assert cls.shape[0] == 0


def test_mosaic4_center_outside_canvas_does_not_raise():
    s = 64
    img = np.zeros((s, s, 3), dtype=np.uint8)
    boxes = np.array([[1.0, 1.0, 10.0, 10.0]], dtype=np.float32)
    cls = np.array([0], dtype=np.int32)
    for center in ((-s // 2, -s // 2), (2 * s + s // 2, 2 * s + s // 2)):
        out, xyxy, _ = mosaic4(
            [img, img, img, img],
            [boxes, boxes, boxes, boxes],
            [cls, cls, cls, cls],
            imgsz=s,
            center=center,
        )
        assert out.shape == (2 * s, 2 * s, 3)
