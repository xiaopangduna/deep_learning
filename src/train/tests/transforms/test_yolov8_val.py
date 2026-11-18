"""验证预处理与 Ultralytics 8.3.189 ``YOLO.val()`` 的几何、像素一致。"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.data.augment import LetterBox
from ultralytics.utils.instance import Instances

from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset
from lovely_deep_learning.transforms.yolov8_val import (
    YOLOv8ValAugment,
    official_val_bgr,
    rectangular_val_shapes,
)


def _ultralytics_val_bgr(
    bgr: np.ndarray,
    rel_xywh: np.ndarray,
    dst_hw: tuple[int, int],
    imgsz: int = 640,
) -> tuple[np.ndarray, np.ndarray]:
    """复刻 ``load_image(rect_mode=True)`` + ``LetterBox(scaleup=False)``。"""
    h0, w0 = bgr.shape[:2]
    r = imgsz / max(h0, w0)
    im = bgr
    if r != 1:
        w = min(math.ceil(w0 * r), imgsz)
        h = min(math.ceil(h0 * r), imgsz)
        im = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    inst = Instances(
        np.asarray(rel_xywh, dtype=np.float32).reshape(-1, 4),
        segments=np.zeros((0, 1000, 2), dtype=np.float32),
        bbox_format="xywh",
        normalized=True,
    )
    out = LetterBox(new_shape=(int(dst_hw[0]), int(dst_hw[1])), scaleup=False)(
        {"img": im, "instances": inst}
    )
    return out["img"], np.asarray(out["instances"].bboxes, dtype=np.float32)


def _xyxy_on_original(rel_xywh: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return ObjectDetectDataset.convert_bboxes_from_cxcywh_relative_to_xyxy_absolute(
        np.asarray(rel_xywh, dtype=np.float32).reshape(-1, 4), hw
    )


def test_rectangular_val_shapes_match_ultralytics_formula():
    hw = np.array(
        [[480, 640], [100, 100], [640, 480], [200, 400]],
        dtype=np.float64,
    )
    order, dst = rectangular_val_shapes(hw, batch_size=2, imgsz=640, stride=32, pad=0.5)
    # 宽高比 0.5, 0.75, 1, 4/3 → 下标 3, 0, 1, 2
    np.testing.assert_array_equal(order, [3, 0, 1, 2])
    np.testing.assert_array_equal(dst[0], [512, 672])
    np.testing.assert_array_equal(dst[1], [512, 672])
    np.testing.assert_array_equal(dst[2], [672, 672])
    np.testing.assert_array_equal(dst[3], [672, 672])


def test_official_val_bgr_matches_ultralytics_letterbox():
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(333, 500, 3), dtype=np.uint8)
    rel = np.array([[0.5, 0.4, 0.2, 0.3]], dtype=np.float32)
    xyxy = _xyxy_on_original(rel, (333, 500))
    for dst in ((640, 640), (512, 672), (256, 640)):
        ours_bgr, ours_xyxy = official_val_bgr(bgr, xyxy, dst, imgsz=640)
        ref_bgr, ref_xyxy = _ultralytics_val_bgr(bgr, rel, dst, imgsz=640)
        assert ours_bgr.shape == ref_bgr.shape
        assert ours_bgr.dtype == np.uint8
        np.testing.assert_array_equal(ours_bgr, ref_bgr)
        np.testing.assert_allclose(ours_xyxy, ref_xyxy, atol=1e-4, rtol=0.0)


def test_val_augment_rect_batches_share_shape(tmp_path):
    sizes = [(480, 640), (100, 100), (640, 480), (200, 400)]
    rows = ["path_img,path_label_detect_yolo"]
    for i, (h, w) in enumerate(sizes):
        img_path = tmp_path / f"{i}.jpg"
        label_path = tmp_path / f"{i}.txt"
        cv2.imwrite(str(img_path), np.full((h, w, 3), 30 + i, dtype=np.uint8))
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        rows.append(f"{img_path},{label_path}")
    csv_path = tmp_path / "val.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = ObjectDetectDataset(
        csv_paths=[csv_path],
        key_map={"img_path": "path_img", "object_label_path": "path_label_detect_yolo"},
        transform=YOLOv8ValAugment(imgsz=640, rect=True, stride=32, pad=0.5),
        batch_size=2,
    )
    assert len(dataset) == 4
    shapes = []
    for i in range(4):
        net_in, net_out = dataset[i]
        img = net_in["img_tv_transformed"]
        assert img.dtype == torch.uint8
        shapes.append(tuple(img.shape[-2:]))
        boxes = net_out["bboxes_xyxy_abs_tv_transformed"]
        assert tuple(boxes.canvas_size) == shapes[-1]
    assert shapes[0] == shapes[1] == (512, 672)
    assert shapes[2] == shapes[3] == (672, 672)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dataset.get_collate_fn_for_dataloader(),
    )
    net_in, _net_out = next(iter(loader))
    stacked = torch.stack([sample["img"] for sample in net_in], dim=0)
    assert tuple(stacked.shape) == (2, 3, 512, 672)
