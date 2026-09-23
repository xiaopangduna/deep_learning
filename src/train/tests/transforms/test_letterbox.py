"""LetterBox：非正方形图 pad 到 640，框随 pad 平移。"""

from __future__ import annotations

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose, ToDtype

from lovely_deep_learning.transforms.letterbox import LetterBox, letterbox_resize_and_pad


def test_letterbox_resize_and_pad_landscape_480x640():
    """高 480 宽 640：r=1，上下各 pad 80。"""
    rh, rw, left, top, right, bottom = letterbox_resize_and_pad(
        480, 640, 640, 640
    )
    assert (rh, rw) == (480, 640)
    assert (left, right) == (0, 0)
    assert (top, bottom) == (80, 80)


def test_letterbox_resize_and_pad_portrait_640x480():
    """高 640 宽 480：左右各 pad 80。"""
    rh, rw, left, top, right, bottom = letterbox_resize_and_pad(
        640, 480, 640, 640
    )
    assert (rh, rw) == (640, 480)
    assert (left, right) == (80, 80)
    assert (top, bottom) == (0, 0)


def test_letterbox_image_only_shape():
    img = tv_tensors.Image(torch.zeros(3, 480, 640, dtype=torch.uint8))
    out = LetterBox(size=[640, 640])(img)
    assert tuple(out.shape) == (3, 640, 640)
    assert out.dtype == torch.uint8
    assert int(out[:, 0, 0].max()) == 114
    assert int(out[:, 80, 0].max()) == 0


def test_letterbox_boxes_shift_with_vertical_pad():
    img = tv_tensors.Image(torch.zeros(3, 480, 640, dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        format="XYXY",
        canvas_size=(480, 640),
    )
    img_out, boxes_out = LetterBox(size=[640, 640])(img, boxes)
    assert tuple(img_out.shape) == (3, 640, 640)
    assert tuple(boxes_out.canvas_size) == (640, 640)
    torch.testing.assert_close(
        boxes_out.as_subclass(torch.Tensor),
        torch.tensor([[10.0, 90.0, 50.0, 130.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_letterbox_boxes_shift_with_horizontal_pad():
    img = tv_tensors.Image(torch.zeros(3, 640, 480, dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        format="XYXY",
        canvas_size=(640, 480),
    )
    _img_out, boxes_out = LetterBox(size=[640, 640])(img, boxes)
    torch.testing.assert_close(
        boxes_out.as_subclass(torch.Tensor),
        torch.tensor([[90.0, 10.0, 130.0, 50.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_letterbox_in_compose_with_target_dict():
    tf = Compose(
        [
            LetterBox(size=[640, 640], fill=114),
            ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    img = tv_tensors.Image(torch.zeros(3, 480, 640, dtype=torch.uint8))
    target = {
        "cls": torch.tensor([3], dtype=torch.int32),
        "bboxes": tv_tensors.BoundingBoxes(
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format="XYXY",
            canvas_size=(480, 640),
        ),
    }
    img_out, target_out = tf(img, target)
    assert tuple(img_out.shape) == (3, 640, 640)
    assert img_out.dtype == torch.float32
    assert int(target_out["cls"][0]) == 3
    torch.testing.assert_close(
        target_out["bboxes"].as_subclass(torch.Tensor),
        torch.tensor([[10.0, 90.0, 50.0, 130.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_letterbox_skip_if_size_keeps_mosaic_canvas():
    img = tv_tensors.Image(torch.zeros(3, 1280, 1280, dtype=torch.uint8))
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([[400.0, 400.0, 800.0, 800.0]]),
        format="XYXY",
        canvas_size=(1280, 1280),
    )
    img_out, boxes_out = LetterBox(
        size=[640, 640], skip_if_size=[1280, 1280]
    )(img, boxes)
    assert tuple(img_out.shape) == (3, 1280, 1280)
    torch.testing.assert_close(
        boxes_out.as_subclass(torch.Tensor),
        torch.tensor([[400.0, 400.0, 800.0, 800.0]]),
        atol=0.0,
        rtol=0.0,
    )


def test_mosaic_canvas_affine_then_center_crop_boxes():
    """1280 画布中心框，无平移旋转时裁 640 后落在 [80,80,480,480]。"""
    from torchvision.transforms.v2 import CenterCrop, Compose, RandomAffine

    tf = Compose(
        [
            LetterBox(size=[640, 640], skip_if_size=[1280, 1280]),
            RandomAffine(
                degrees=[0.0, 0.0],
                translate=[0.0, 0.0],
                scale=[1.0, 1.0],
                fill=114,
            ),
            CenterCrop(size=[640, 640]),
        ]
    )
    img = tv_tensors.Image(torch.zeros(3, 1280, 1280, dtype=torch.uint8))
    target = {
        "cls": torch.tensor([1], dtype=torch.int32),
        "bboxes": tv_tensors.BoundingBoxes(
            torch.tensor([[400.0, 400.0, 800.0, 800.0]]),
            format="XYXY",
            canvas_size=(1280, 1280),
        ),
    }
    img_out, target_out = tf(img, target)
    assert tuple(img_out.shape) == (3, 640, 640)
    torch.testing.assert_close(
        target_out["bboxes"].as_subclass(torch.Tensor),
        torch.tensor([[80.0, 80.0, 480.0, 480.0]]),
        atol=1e-4,
        rtol=0.0,
    )
