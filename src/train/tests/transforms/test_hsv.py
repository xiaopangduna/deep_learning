"""RandomHSV：零增益恒等；非零增益改像素、不改框。"""

from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose

from lovely_deep_learning.transforms.hsv import RandomHSV


def test_random_hsv_zero_gain_identity():
    img = tv_tensors.Image(torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8))
    out = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.0)(img.clone())
    torch.testing.assert_close(out, img)


def test_random_hsv_keeps_boxes():
    img = tv_tensors.Image(torch.randint(1, 255, (3, 48, 48), dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[4.0, 5.0, 20.0, 22.0]]),
        format="XYXY",
        canvas_size=(48, 48),
    )
    target = {"cls": torch.tensor([3], dtype=torch.int32), "bboxes": boxes}
    tf = Compose([RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4)])
    img_out, target_out = tf(img, target)
    assert tuple(img_out.shape) == (3, 48, 48)
    torch.testing.assert_close(
        target_out["bboxes"].as_subclass(torch.Tensor),
        boxes.as_subclass(torch.Tensor),
    )
    assert int(target_out["cls"][0]) == 3
