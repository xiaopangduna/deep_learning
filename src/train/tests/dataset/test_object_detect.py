import pytest
from pathlib import Path
import cv2
import torch
import numpy as np
from torchvision import tv_tensors
from torchvision.transforms import v2
from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset


PATH_CSV = ["tests/test_data/coco8/train.csv"]
PATH_CSV_WITHOUT_LABEL = [
    "tests/test_data/coco8/predict.csv"]
KEY_MAP = {"img_path": "path_img",
           "object_label_path": "path_label_detect_yolo"}
PREDICT_KEY_MAP = {"img_path": "path_img"}
MAP_CLASS_ID_TO_CLASS_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}


def test_ObjectDetectDataset_init():
    dataset = ObjectDetectDataset(csv_paths=PATH_CSV, key_map=KEY_MAP,
                                  transform=None, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    assert len(dataset.sample_path_table) == 4
    assert "img_path" in dataset.sample_path_table.columns
    assert "object_label_path" in dataset.sample_path_table.columns


def test_ObjectDetectDataset_init_without_label():
    dataset = ObjectDetectDataset(csv_paths=PATH_CSV_WITHOUT_LABEL, key_map=PREDICT_KEY_MAP,
                                  transform=None, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    assert len(dataset.sample_path_table) == 4
    assert "img_path" in dataset.sample_path_table.columns
    assert "object_label_path" not in dataset.sample_path_table.columns


def test_ImageClassifierDataset_getitem_with_transform():
    transforms = v2.Compose(
        [v2.Resize(size=(640, 640)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ObjectDetectDataset(csv_paths=PATH_CSV, key_map=KEY_MAP,
                                  transform=transforms, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    assert net_in["img_tv_transformed"].shape == (3, 640, 640)


def test_ObjectDetectDataset_getitem_with_transform_without_label():
    transforms = v2.Compose(
        [v2.Resize(size=(640, 640)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ObjectDetectDataset(csv_paths=PATH_CSV_WITHOUT_LABEL, key_map=PREDICT_KEY_MAP,
                                  transform=transforms, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    assert net_in["img_tv_transformed"].shape == (3, 640, 640)


def test_ObjectDetectDataset_draw_label_on_numpy():
    expected_bboxes_xyxy = np.array(
        [[1.0, 20.0, 442.0, 399.0]], dtype=np.float32)

    dataset = ObjectDetectDataset(csv_paths=PATH_CSV, key_map=KEY_MAP,
                                  transform=None, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    img_np = cv2.imread(net_in["img_path"])
    bboxes_xyxy = net_out["bboxes_xyxy_abs_tv_transformed"].numpy()
    cls_np = net_out["cls_np"]
    img_with_label = dataset.draw_label_on_numpy(img_np, bboxes_xyxy, cls_np)
    np.testing.assert_allclose(
        bboxes_xyxy,
        expected_bboxes_xyxy,
        atol=2.0,
        rtol=0.0,
        err_msg="像素 XYXY 与期望值逐元素绝对差应 ≤2",
    )
    cv2.imwrite(
        "./tmp/test_ObjectDetectDataset_draw_label_on_numpy.jpg", img_with_label)


def test_ObjectDetectDataset_draw_target_and_predict_label_on_numpy():
    """左右拼接：左预测、右真值；预测暂与真值使用同一组框与类别。"""
    expected_bboxes_xyxy = np.array(
        [[1.0, 20.0, 442.0, 399.0]], dtype=np.float32)

    dataset = ObjectDetectDataset(csv_paths=PATH_CSV, key_map=KEY_MAP,
                                  transform=None, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    img_np = cv2.imread(net_in["img_path"])
    bboxes_xyxy = net_out["bboxes_xyxy_abs_tv_transformed"].numpy()
    cls_np = net_out["cls_np"]
    np.testing.assert_allclose(
        bboxes_xyxy,
        expected_bboxes_xyxy,
        atol=2.0,
        rtol=0.0,
        err_msg="像素 XYXY 与期望值逐元素绝对差应 ≤2",
    )
    pred_scores = np.ones(bboxes_xyxy.shape[0], dtype=np.float32)
    img_pair = ObjectDetectDataset.draw_target_and_predict_label_on_numpy(
        img_np,
        bboxes_xyxy,
        cls_np,
        bboxes_xyxy,
        cls_np,
        class_names=MAP_CLASS_ID_TO_CLASS_NAME,
        pred_scores=pred_scores,
    )
    h, w = img_np.shape[:2]
    gap = 3
    assert img_pair.shape == (h, 2 * w + gap, 3)
    cv2.imwrite(
        "./tmp/test_ObjectDetectDataset_draw_target_and_predict_label_on_numpy.jpg",
        img_pair,
    )


def test_ObjectDetectDataset_draw_label_on_numpy_with_transform():
    expected_bboxes_xyxy = np.array(
        [[1.0, 30.0, 442.0, 601.0]], dtype=np.float32)

    transforms = v2.Compose(
        [v2.Resize(size=(640, 640)), v2.ToDtype(dtype=torch.float32, scale=True)])
    dataset = ObjectDetectDataset(csv_paths=PATH_CSV, key_map=KEY_MAP,
                                  transform=transforms, map_class_id_to_class_name=MAP_CLASS_ID_TO_CLASS_NAME)
    net_in, net_out = dataset[0]
    img_tensor = net_in["img_tv_transformed"]
    img_np = dataset.convert_img_from_tensor_to_numpy(img_tensor)
    bboxes_xyxy = net_out["bboxes_xyxy_abs_tv_transformed"].numpy()
    cls_np = net_out["cls_tv_transformed"].numpy()

    img_with_label = dataset.draw_label_on_numpy(img_np, bboxes_xyxy, cls_np)
    assert img_with_label.shape == (640, 640, 3)
    np.testing.assert_allclose(
        bboxes_xyxy,
        expected_bboxes_xyxy,
        atol=2.0,
        rtol=0.0,
        err_msg="像素 XYXY 与期望值逐元素绝对差应 ≤2",
    )
    cv2.imwrite(
        "./tmp/test_ObjectDetectDataset_draw_label_on_numpy_with_transform.jpg", img_with_label)
