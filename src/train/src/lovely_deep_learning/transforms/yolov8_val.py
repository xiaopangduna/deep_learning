"""YOLOv8 验证预处理，对齐 Ultralytics ``YOLO.val()``（8.3.189）。

官方验证没有颜色和几何增强。每张图两步：

1. ``load_image(rect_mode=True)``：``r = imgsz / max(h, w)``，长边用 ``ceil`` 缩到
   ``imgsz``，小图也会放大。插值是 ``cv2.INTER_LINEAR``。
2. ``LetterBox(scaleup=False)``：不再放大，只 pad（目标比当前图小时才会再缩小）。
   居中 pad 用 ``round(d ± 0.1)``，填充 114。框按 LetterBox 的浮点 ``r`` 缩放，
   而不是按取整后的宽高比。

``rect=True``（``model.val()`` 的默认值）时，样本按原始高宽比排序，每个 batch
共用一个目标形状：

``ceil(shape * imgsz / stride + pad) * stride``，``pad=0.5``。

因此验证输入经常不是 640×640，长边常是 672。同一 batch 内形状相同，才能
``torch.stack``。``rect`` 分组用 Dataset 的 ``batch_size``，必须和 DataLoader 一致，
且验证不能 shuffle。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

from .mosaic import resize_long_side


def rectangular_val_shapes(
    hw: np.ndarray,
    batch_size: int,
    imgsz: int,
    stride: int = 32,
    pad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """返回按宽高比排序的 ``order``，以及排序后每张图的目标 ``(h, w)``。

    与 ``ultralytics.data.base.BaseDataset.set_rectangle`` 同一公式。``hw`` 是原始
    高宽，形状 ``(N, 2)``。
    """
    hw = np.asarray(hw, dtype=np.float64)
    n = int(hw.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    batch_size = max(int(batch_size), 1)
    ar = hw[:, 0] / np.maximum(hw[:, 1], 1.0)
    order = np.argsort(ar, kind="quicksort")
    ar_sorted = ar[order]
    bi = np.floor(np.arange(n) / batch_size).astype(np.int64)
    nb = int(bi[-1]) + 1
    shapes = np.ones((nb, 2), dtype=np.float64)
    for i in range(nb):
        ari = ar_sorted[bi == i]
        mini = float(ari.min())
        maxi = float(ari.max())
        if maxi < 1:
            shapes[i] = (maxi, 1.0)
        elif mini > 1:
            shapes[i] = (1.0, 1.0 / mini)
    batch_shapes = np.ceil(shapes * float(imgsz) / float(stride) + float(pad)).astype(np.int64) * int(stride)
    return order.astype(np.int64), batch_shapes[bi]


def letterbox_scaleup_false(
    img: np.ndarray,
    boxes: np.ndarray,
    dst_hw: tuple[int, int],
    fill: int = 114,
) -> tuple[np.ndarray, np.ndarray]:
    """``LetterBox(scaleup=False, center=True)``。框乘浮点 ``r``，再加整数 pad。"""
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    r = min(dst_h / src_h, dst_w / src_w)
    r = min(r, 1.0)
    new_w = int(round(src_w * r))
    new_h = int(round(src_h * r))
    dw = float(dst_w - new_w)
    dh = float(dst_h - new_h)
    dw /= 2.0
    dh /= 2.0
    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        boxes[:, [0, 2]] *= r
        boxes[:, [1, 3]] *= r
    if (src_h, src_w) != (new_h, new_w):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(int(fill),) * 3
    )
    if boxes.size:
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
    return img, boxes


def official_val_bgr(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    dst_hw: tuple[int, int],
    imgsz: int,
    fill: int = 114,
) -> tuple[np.ndarray, np.ndarray]:
    """先长边 ``ceil`` 到 ``imgsz``，再 ``LetterBox(scaleup=False)`` 到 ``dst_hw``。"""
    img, boxes = resize_long_side(img, boxes_xyxy, int(imgsz))
    return letterbox_scaleup_false(img, boxes, dst_hw, fill=fill)


def _image_hw(path: str) -> tuple[int, int]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return int(img.shape[0]), int(img.shape[1])


class YOLOv8ValAugment:
    """官方验证预处理。``reads_dataset`` 为真时 Dataset 以 ``transform(index, dataset)`` 调用。

    返回 uint8 RGB ``(3, H, W)``。``/255`` 由 ``model_float_images`` 完成。
    """

    reads_dataset = True

    def __init__(
        self,
        imgsz: int = 640,
        rect: bool = True,
        stride: int = 32,
        pad: float = 0.5,
        fill: int = 114,
    ) -> None:
        self.imgsz = int(imgsz)
        self.rect = bool(rect)
        self.stride = int(stride)
        self.pad = float(pad)
        self.fill = int(fill)
        self._dataset: Any = None

    def bind(self, dataset: Any) -> None:
        """按原始高宽比排序（``rect`` 时），并把每张图的目标高宽记在 Dataset 上。

        验证和测试可以共用同一个变换对象，目标形状写在各自的 Dataset 上。
        """
        self._dataset = dataset
        if not getattr(dataset, "_has_label", False):
            return
        dataset.prepare_label_cache()
        paths = dataset._img_paths
        n = len(paths)
        if n == 0:
            dataset.yolo_val_dst_hw = np.zeros((0, 2), dtype=np.int64)
            return
        hw = np.asarray([_image_hw(p) for p in paths], dtype=np.float64)
        if self.rect:
            order, dst = rectangular_val_shapes(
                hw,
                batch_size=dataset.batch_size,
                imgsz=self.imgsz,
                stride=self.stride,
                pad=self.pad,
            )
            dataset.reorder_samples(order)
        else:
            dst = np.full((n, 2), self.imgsz, dtype=np.int64)
        dataset.yolo_val_dst_hw = dst

    def __call__(self, index: Any, dataset: Any = None):
        """有标签时 ``index`` 是样本下标；无标签预测传入 RGB uint8 图，按正方形 letterbox。"""
        if not isinstance(index, (int, np.integer)):
            return self._letterbox_tensor(index)
        dataset = self._dataset if dataset is None else dataset
        if dataset is None:
            raise RuntimeError("YOLOv8ValAugment 须先 bind 到 ObjectDetectDataset")
        dst = getattr(dataset, "yolo_val_dst_hw", None)
        if dst is None:
            raise RuntimeError("YOLOv8ValAugment.bind 未写 yolo_val_dst_hw")
        index = int(index)
        path = dataset._img_paths[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        _cls, rel = dataset._labels[index]
        h0, w0 = int(img.shape[0]), int(img.shape[1])
        if rel.size == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
        else:
            from ..dataset.object_detect import ObjectDetectDataset

            xyxy = ObjectDetectDataset.convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
                rel, (h0, w0)
            )
        dst_h, dst_w = int(dst[index, 0]), int(dst[index, 1])
        img, xyxy = official_val_bgr(img, xyxy, (dst_h, dst_w), self.imgsz, self.fill)
        return _pack_rgb(img, xyxy, dataset._labels[index][0])

    def _letterbox_tensor(self, img_tv: torch.Tensor) -> torch.Tensor:
        """无标签预测：长边 ceil 到 ``imgsz`` 后 pad 成正方形。"""
        chw = img_tv.detach().cpu()
        if chw.dtype != torch.uint8:
            raise TypeError("无标签验证变换期望 uint8 RGB (C, H, W)")
        rgb = chw.permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = np.zeros((0, 4), dtype=np.float32)
        bgr, _boxes = official_val_bgr(
            bgr, boxes, (self.imgsz, self.imgsz), self.imgsz, self.fill
        )
        out_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return tv_tensors.Image(torch.from_numpy(out_rgb).permute(2, 0, 1).contiguous())


def _pack_rgb(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    cls: np.ndarray,
) -> tuple[tv_tensors.Image, torch.Tensor, tv_tensors.BoundingBoxes]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_tv = tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    out_h, out_w = int(rgb.shape[0]), int(rgb.shape[1])
    bboxes_tv = tv_tensors.BoundingBoxes(
        torch.from_numpy(np.ascontiguousarray(xyxy, dtype=np.float32)),
        format="XYXY",
        canvas_size=(out_h, out_w),
    )
    cls_t = torch.from_numpy(np.asarray(cls, dtype=np.int32).copy())
    return img_tv, cls_t, bboxes_tv
