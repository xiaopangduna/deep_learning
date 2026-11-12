"""YOLOv8 训练增强：Mosaic、一次仿射、HSV、左右翻转，整段放在 ``transform_train``。

构造时还没有 Dataset。``ObjectDetectDataset`` 创建后调用 ``bind``，Mosaic 再通过
``load_raw`` 取另外三张已缩放的图。不要调用 ``dataset[index]``，那会再次进入本变换。

几何在 numpy 里做完，返回 uint8 RGB ``(3, s, s)``。``/255`` 由后续 ``model_float_images`` 在 GPU 上完成。
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import tv_tensors

from .mosaic import mosaic4
from .yolo_aug import hsv_bgr_, letterbox_bgr, maybe_fliplr, random_affine


class YOLOv8TrainAugment:
    """官方 hyp 对应的一条训练管道。``reads_dataset`` 为真时 Dataset 以 ``transform(index)`` 调用。"""

    reads_dataset = True

    def __init__(
        self,
        mosaic_prob: float = 1.0,
        mosaic_size: int = 640,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        fliplr: float = 0.5,
    ) -> None:
        self.mosaic_prob = float(mosaic_prob)
        self.mosaic_size = int(mosaic_size)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.fliplr = float(fliplr)
        self._dataset: Any = None

    def bind(self, dataset: Any) -> None:
        """记下 Dataset，并按 ``mosaic_size`` 准备缩放图缓冲与标签缓存。"""
        self._dataset = dataset
        dataset.prepare_train_cache(self.mosaic_size)

    def __call__(self, index: int):
        """返回 uint8 RGB 图、类别张量、像素 XYXY 框。"""
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("YOLOv8TrainAugment 须先 bind 到 ObjectDetectDataset")
        n = max(len(dataset), 1)
        s = self.mosaic_size
        use_mosaic = self.mosaic_prob > 0.0 and random.random() < self.mosaic_prob
        if use_mosaic:
            indices = [index] + [random.randint(0, n - 1) for _ in range(3)]
            images: list[np.ndarray] = []
            boxes_list: list[np.ndarray] = []
            cls_list: list[np.ndarray] = []
            for j in indices:
                im, _hw, cls_j, xyxy_j = dataset.load_raw(j)
                images.append(im)
                boxes_list.append(xyxy_j)
                cls_list.append(cls_j)
            canvas, xyxy, cls = mosaic4(images, boxes_list, cls_list, imgsz=s)
            img, xyxy, cls = random_affine(
                canvas,
                xyxy,
                cls,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                border=(-s // 2, -s // 2),
            )
        else:
            im, _hw, cls, xyxy = dataset.load_raw(index)
            img, xyxy = letterbox_bgr(im, xyxy, s)
            img, xyxy, cls = random_affine(
                img,
                xyxy,
                cls,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                border=(0, 0),
            )
        hsv_bgr_(img, self.hsv_h, self.hsv_s, self.hsv_v)
        img, xyxy = maybe_fliplr(img, xyxy, self.fliplr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tv = tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
        out_h, out_w = int(rgb.shape[0]), int(rgb.shape[1])
        bboxes_tv = tv_tensors.BoundingBoxes(
            torch.from_numpy(np.ascontiguousarray(xyxy, dtype=np.float32)),
            format="XYXY",
            canvas_size=(out_h, out_w),
        )
        cls_t = torch.from_numpy(np.asarray(cls, dtype=np.int32))
        return img_tv, cls_t, bboxes_tv
