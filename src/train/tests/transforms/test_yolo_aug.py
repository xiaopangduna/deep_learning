"""numpy 训练增强：Mosaic border 一次仿射得到目标边长。"""

import random

import numpy as np
import torch

from lovely_deep_learning.dataset.base import BaseDataset
from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset
from lovely_deep_learning.transforms.yolo_aug import random_affine


def test_affine_zero_params_keeps_pixels():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:16, 8:16] = 200
    boxes = np.array([[8, 8, 16, 16]], dtype=np.float32)
    cls = np.array([3], dtype=np.int32)
    random.seed(0)
    out, new_boxes, new_cls = random_affine(
        img, boxes, cls, degrees=0.0, translate=0.0, scale=0.0, border=(0, 0)
    )
    assert out.shape == img.shape
    assert np.array_equal(out, img)
    assert new_cls.tolist() == [3]
    assert new_boxes.shape == (1, 4)
    assert np.allclose(new_boxes, boxes, atol=1e-3)


def test_mosaic_border_warp_outputs_half_canvas():
    canvas = np.full((128, 128, 3), 10, dtype=np.uint8)
    boxes = np.array([[4, 4, 40, 40]], dtype=np.float32)
    cls = np.array([1], dtype=np.int32)
    random.seed(1)
    out, new_boxes, new_cls = random_affine(
        canvas,
        boxes,
        cls,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        border=(-32, -32),
    )
    assert out.shape == (64, 64, 3)
    assert new_boxes.shape[0] == new_cls.shape[0]
    if new_boxes.size:
        assert (new_boxes[:, 0] >= 0).all()
        assert (new_boxes[:, 2] <= 64).all()


def test_numpy_aug_dataset_returns_uint8_square():
    from lovely_deep_learning.transforms.yolov8_train import YOLOv8TrainAugment

    dataset = ObjectDetectDataset(
        csv_paths=["tests/test_data/coco8/train.csv"],
        key_map={"img_path": "path_img", "object_label_path": "path_label_detect_yolo"},
        transform=YOLOv8TrainAugment(
            mosaic_prob=1.0,
            mosaic_size=64,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
        ),
        batch_size=2,
    )
    net_in, net_out = dataset[0]
    img = net_in["img_tv_transformed"]
    assert img.dtype == torch.uint8
    assert tuple(img.shape) == (3, 64, 64)
    assert int(img.max()) <= 255
    boxes = net_out["bboxes_xyxy_abs_tv_transformed"]
    assert tuple(boxes.canvas_size) == (64, 64)


def test_model_float_images_scales_uint8():
    raw = torch.tensor([0, 255], dtype=torch.uint8).reshape(1, 1, 1, 2)
    out = BaseDataset.model_float_images(raw)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2))
    already = torch.tensor([0.5])
    assert BaseDataset.model_float_images(already) is already
