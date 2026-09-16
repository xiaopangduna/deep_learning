"""YOLOv8 风格 RandomHSV：在 uint8 RGB 上用 OpenCV LUT 调 H/S/V。

增益默认 ``0.015 / 0.7 / 0.4``，与 Ultralytics YOLOv8 超参 ``hsv_h/s/v`` 一致。
Hue LUT 用较新官方实现 ``(x + r*180) % 180``，并把 ``sat[0]=0`` 以免纯白偏色。
须放在 ``ToDtype`` 之前。不改 BoundingBoxes。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform


class RandomHSV(Transform):
    """随机 HSV；``hgain/sgain/vgain`` 为相对幅度，采样 ``U(-1,1)*gain``。"""

    def __init__(
        self,
        hgain: float = 0.015,
        sgain: float = 0.7,
        vgain: float = 0.4,
    ) -> None:
        super().__init__()
        self.hgain = float(hgain)
        self.sgain = float(sgain)
        self.vgain = float(vgain)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt
        if not (torch.is_tensor(inpt) and inpt.ndim >= 3 and int(inpt.shape[-3]) == 3):
            return inpt
        if not (self.hgain or self.sgain or self.vgain):
            return inpt

        img = inpt
        leading = img.shape[:-3]
        c, h, w = int(img.shape[-3]), int(img.shape[-2]), int(img.shape[-1])
        flat = img.reshape(-1, c, h, w)
        outs = []
        for i in range(flat.shape[0]):
            outs.append(self._hsv_one(flat[i]))
        stacked = torch.stack(outs, dim=0).reshape(*leading, c, h, w)
        if isinstance(inpt, tv_tensors.Image):
            return tv_tensors.wrap(stacked, like=inpt)
        return stacked

    def _hsv_one(self, chw: torch.Tensor) -> torch.Tensor:
        was_float = chw.is_floating_point()
        if was_float:
            np_img = (
                (chw.detach().cpu().clamp(0, 1) * 255.0)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
        else:
            np_img = chw.detach().cpu().to(torch.uint8).permute(1, 2, 0).numpy()

        r = np.random.uniform(-1, 1, 3) * [
            self.hgain,
            self.sgain,
            self.vgain,
        ]
        dtype = np_img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0

        hue, sat, val = cv2.split(cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV))
        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        out_np = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        out = torch.from_numpy(out_np).permute(2, 0, 1).to(device=chw.device)
        if was_float:
            return out.to(dtype=chw.dtype) / 255.0
        return out.to(dtype=chw.dtype)
