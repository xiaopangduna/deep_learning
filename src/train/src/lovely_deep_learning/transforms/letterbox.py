"""YOLOv8 风格 LetterBox：保比例缩放后 pad 到固定 ``(H, W)``。

几何与 ``ultralytics.data.augment.LetterBox``（``auto=False``, ``scale_fill=False``）一致：
``r = min(dst_h/src_h, dst_w/src_w)``，``round`` 得到未 pad 尺寸，居中时用
``round(d-0.1)`` / ``round(d+0.1)`` 拆分左右/上下，避免奇数 pad 对不齐。

官方 ``YOLO.val()`` 的 ``LetterBox(scaleup=False)`` 只负责 pad；在那之前
``load_image(rect_mode=True)`` 已经把长边缩放到 ``imgsz``（**含放大**）。要对齐官方
val mAP，这里的 ``scaleup`` 应保持默认 ``True``。

设计为 :class:`torchvision.transforms.v2.Transform`，可放进 ``v2.Compose``：
对 ``Image`` / ``BoundingBoxes`` 使用同一套 resize+pad（框随 pad 平移）。
须放在 ``ToDtype`` **之前**（默认 ``fill=114`` 按 uint8）。
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import torch
from torchvision.transforms.v2 import InterpolationMode, Transform, functional as F
from torchvision.transforms.v2._utils import _get_fill, _setup_fill_arg, query_size
from torchvision.transforms.v2.functional._utils import _FillType


def _as_hw(size: Union[int, Sequence[int]]) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"size 须为 int 或长度为 2 的序列 (H, W)，收到 {size!r}")
    return int(size[0]), int(size[1])


def letterbox_resize_and_pad(
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    *,
    scaleup: bool = True,
    center: bool = True,
) -> tuple[int, int, int, int, int, int]:
    """返回 ``(resize_h, resize_w, left, top, right, bottom)``，与官方 LetterBox 取整一致。"""
    r = min(dst_h / src_h, dst_w / src_w)
    if not scaleup:
        r = min(r, 1.0)
    resize_w = int(round(src_w * r))
    resize_h = int(round(src_h * r))
    dw = float(dst_w - resize_w)
    dh = float(dst_h - resize_h)
    if center:
        dw /= 2.0
        dh /= 2.0
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
    else:
        top, left = 0, 0
        bottom = int(round(dh + 0.1))
        right = int(round(dw + 0.1))
    return resize_h, resize_w, left, top, right, bottom


class LetterBox(Transform):
    """保比例缩放到不超过 ``size``，再 pad 到 ``size``（默认灰边 114）。"""

    def __init__(
        self,
        size: Union[int, Sequence[int]] = 640,
        fill: Union[_FillType, dict[Union[type, str], _FillType]] = 114,
        center: bool = True,
        scaleup: bool = True,
        interpolation: Union[int, InterpolationMode] = 2,
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size = _as_hw(size)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)
        self.center = bool(center)
        self.scaleup = bool(scaleup)
        self.interpolation = interpolation
        self.antialias = bool(antialias)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        src_h, src_w = query_size(flat_inputs)
        dst_h, dst_w = self.size
        resize_h, resize_w, left, top, right, bottom = letterbox_resize_and_pad(
            src_h,
            src_w,
            dst_h,
            dst_w,
            scaleup=self.scaleup,
            center=self.center,
        )
        return {
            "resize_size": [resize_h, resize_w],
            "padding": [left, top, right, bottom],
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        if (
            torch.is_tensor(inpt)
            and inpt.is_floating_point()
            and isinstance(fill, (int, float))
            and fill > 1
        ):
            fill = float(fill) / 255.0
        out = self._call_kernel(
            F.resize,
            inpt,
            size=params["resize_size"],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        return self._call_kernel(
            F.pad,
            out,
            padding=params["padding"],
            fill=fill,
            padding_mode="constant",
        )
