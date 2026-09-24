import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

from lovely_deep_learning.dataset.box_crop_classifier import BoxCropClassifierDataset

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "yolo_to_box_crop_csv.py"
_spec = importlib.util.spec_from_file_location("yolo_to_box_crop_csv", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

COCO8 = Path(__file__).resolve().parents[2] / "datasets" / "COCO8"
PERSON_CAR = {0: "person", 1: "car"}


def test_csv_keeps_every_class_and_image_size(tmp_path):
    _mod.convert(COCO8 / "train.csv", COCO8 / "val.csv", tmp_path)
    val = pd.read_csv(tmp_path / "val.csv")
    assert list(val.columns) == _mod.LABEL_COLUMNS
    assert set(val["class_name"]) - {"person", "car"}
    person = val[val["class_name"] == "person"]
    assert (person["class_id"] == 0).all()
    predict = pd.read_csv(tmp_path / "predict.csv")
    assert list(predict.columns) == _mod.PREDICT_COLUMNS
    assert len(predict) == len(val)


def test_dataset_filters_class_and_short_side(tmp_path):
    _mod.convert(COCO8 / "train.csv", COCO8 / "val.csv", tmp_path)
    raw = pd.read_csv(tmp_path / "val.csv")
    labeled = BoxCropClassifierDataset(
        [tmp_path / "val.csv"],
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=16,
    )
    assert len(labeled) < len(raw)
    assert set(labeled.sample_path_table["class_name"]) <= {"person", "car"}
    assert set(labeled.sample_path_table["class_id"].astype(int)) <= {0, 1}

    person = raw[raw["class_name"] == "person"]
    short_side = np.minimum(person["w"] * person["img_w"], person["h"] * person["img_h"])
    midpoint = float((short_side.min() + short_side.max()) / 2)
    strict = BoxCropClassifierDataset(
        [tmp_path / "val.csv"],
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=midpoint,
    )
    assert 0 < len(strict) < len(labeled)


def test_box_crop_is_smaller_than_source_and_predict_has_no_label(tmp_path):
    _mod.convert(COCO8 / "train.csv", COCO8 / "val.csv", tmp_path)
    transform = v2.Compose(
        [v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)]
    )
    labeled = BoxCropClassifierDataset(
        [tmp_path / "val.csv"],
        transform=transform,
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=1,
    )
    net_in, net_out = labeled[0]
    assert net_in["img_tv_transformed"].shape == (3, 224, 224)
    assert net_out["class_id"] in (0, 1)
    assert net_out["class_name"] in ("person", "car")
    src = __import__("cv2").imread(net_in["img_path"])
    raw = BoxCropClassifierDataset(
        [tmp_path / "val.csv"],
        transform=None,
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=1,
    )
    crop_in, _ = raw[0]
    crop_h, crop_w = crop_in["img_tv_transformed"].shape[-2:]
    assert crop_h < src.shape[0] or crop_w < src.shape[1]

    predict = BoxCropClassifierDataset(
        [tmp_path / "predict.csv"],
        key_map=BoxCropClassifierDataset.PREDICT_KEY_MAP,
        transform=transform,
        min_side_px=1,
    )
    _, net_out_pred = predict[0]
    assert net_out_pred == {}
