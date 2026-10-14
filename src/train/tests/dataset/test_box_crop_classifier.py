import importlib.util
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import v2

from lovely_deep_learning.data_module.box_crop_classifier import BoxCropClassifierDataModule
from lovely_deep_learning.dataset.image_classifier import (
    ImageClassifierFromDetectBoxDataset,
    max_iou_cxcywh,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "yolo_to_box_crop_csv.py"
_spec = importlib.util.spec_from_file_location("yolo_to_box_crop_csv", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

COCO8 = Path(__file__).resolve().parents[2] / "datasets" / "COCO8"
# COCO8 的 val 里只有 person，原始 id 是 0。
PERSON_CAR = {0: [0]}


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
    labeled = ImageClassifierFromDetectBoxDataset(
        [tmp_path / "val.csv"],
        class_groups=PERSON_CAR,
        min_side_px=16,
    )
    assert len(labeled) < len(raw)
    assert set(labeled.sample_path_table["class_name"]) <= {"person", "car"}
    assert set(labeled.sample_path_table["class_id"].astype(int)) <= {0, 1}

    person = raw[raw["class_name"] == "person"]
    short_side = np.minimum(person["w"] * person["img_w"], person["h"] * person["img_h"])
    midpoint = float((short_side.min() + short_side.max()) / 2)
    strict = ImageClassifierFromDetectBoxDataset(
        [tmp_path / "val.csv"],
        class_groups=PERSON_CAR,
        min_side_px=midpoint,
    )
    assert 0 < len(strict) < len(labeled)


def test_box_crop_is_smaller_than_source_and_predict_has_no_label(tmp_path):
    _mod.convert(COCO8 / "train.csv", COCO8 / "val.csv", tmp_path)
    transform = v2.Compose(
        [v2.Resize(size=(224, 224)), v2.ToDtype(dtype=torch.float32, scale=True)]
    )
    labeled = ImageClassifierFromDetectBoxDataset(
        [tmp_path / "val.csv"],
        transform=transform,
        class_groups=PERSON_CAR,
        min_side_px=1,
    )
    net_in, net_out = labeled[0]
    assert net_in["img_tv_transformed"].shape == (3, 224, 224)
    assert net_out["class_id"] in (0, 1)
    assert net_out["class_name"] in ("person", "car")
    src = __import__("cv2").imread(net_in["img_path"])
    raw = ImageClassifierFromDetectBoxDataset(
        [tmp_path / "val.csv"],
        transform=None,
        class_groups=PERSON_CAR,
        min_side_px=1,
    )
    crop_in, _ = raw[0]
    crop_h, crop_w = crop_in["img_tv_transformed"].shape[-2:]
    assert crop_h < src.shape[0] or crop_w < src.shape[1]

    predict = ImageClassifierFromDetectBoxDataset(
        [tmp_path / "predict.csv"],
        key_map=ImageClassifierFromDetectBoxDataset.PREDICT_KEY_MAP,
        transform=transform,
        min_side_px=1,
    )
    _, net_out_pred = predict[0]
    assert net_out_pred == {}


def _write_age_csv(path: Path) -> None:
    rows = []
    specs = [
        (0, "head-age_0", 40),
        (1, "head-age_1", 40),
        (2, "head-age_2", 40),
        (3, "head-age_3", 40),
        (4, "other", 40),
    ]
    for index, (class_id, class_name, side) in enumerate(specs):
        rows.append(
            {
                "path_img": f"/tmp/img_{index}.jpg",
                "class_name": class_name,
                "class_id": class_id,
                "cx": 0.5,
                "cy": 0.5,
                "w": side / 100,
                "h": side / 100,
                "img_w": 100,
                "img_h": 100,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_class_groups_merge_original_ids_and_drop_the_rest(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    dataset = ImageClassifierFromDetectBoxDataset(
        [csv_path],
        class_groups={0: [0, 1, 3], 1: [2]},
        min_side_px=16,
    )
    table = dataset.sample_path_table
    assert set(table["class_id"].astype(int)) == {0, 1}
    merged = table[table["class_id"].astype(int) == 0]
    assert set(merged["class_name"]) == {"head-age_0+head-age_1+head-age_3"}
    assert (table.loc[table["class_id"].astype(int) == 1, "class_name"] == "head-age_2").all()
    assert "other" not in set(table["class_name"])
    assert dataset.map_class_id_to_class_name == {
        0: "head-age_0+head-age_1+head-age_3",
        1: "head-age_2",
    }
    assert len(dataset) == 4


def test_class_groups_reject_overlap_gap_and_empty_group(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    with pytest.raises(ValueError, match="同时出现"):
        ImageClassifierFromDetectBoxDataset([csv_path], class_groups={0: [0, 1], 1: [1]}, min_side_px=1)
    with pytest.raises(ValueError, match="连续"):
        ImageClassifierFromDetectBoxDataset([csv_path], class_groups={0: [0], 2: [2]}, min_side_px=1)
    with pytest.raises(ValueError, match="没有样本"):
        ImageClassifierFromDetectBoxDataset(
            [csv_path],
            class_groups={0: [0, 1, 3], 1: [2]},
            min_side_px=10_000,
        )


def test_datamodule_passes_class_groups(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    module = BoxCropClassifierDataModule(
        train_csv_paths=[str(csv_path)],
        val_csv_paths=[str(csv_path)],
        test_csv_paths=[str(csv_path)],
        predict_csv_paths=[str(csv_path)],
        transform_train=None,
        transform_val=None,
        class_groups={"0": [0, 1, 3], "1": [2]},
        min_side_px=1,
        batch_size=2,
        num_workers=0,
    )
    module.setup("test")
    assert module.test_dataset.map_class_id_to_class_name[1] == "head-age_2"
    assert len(module.test_dataset) == 4


def _write_box_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _box_row(img_path, class_name, class_id, cx, cy, w, h, img_w=64, img_h=64):
    return {
        "path_img": str(img_path),
        "class_name": class_name,
        "class_id": class_id,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "img_w": img_w,
        "img_h": img_h,
    }


def test_background_ratio_extends_only_the_training_length(tmp_path):
    img_path = tmp_path / "a.jpg"
    cv2.imwrite(str(img_path), np.zeros((64, 64, 3), dtype=np.uint8))
    csv_path = tmp_path / "boxes.csv"
    _write_box_csv(
        csv_path,
        [
            _box_row(img_path, "person", 0, 0.2, 0.2, 0.2, 0.2),
            _box_row(img_path, "dog", 2, 0.8, 0.8, 0.15, 0.15),
        ],
    )
    class_groups = {0: [0], 1: "background"}
    quiet = ImageClassifierFromDetectBoxDataset(
        [csv_path],
        class_groups=class_groups,
        min_side_px=1,
    )
    assert len(quiet) == 1
    assert quiet.map_class_id_to_class_name[1] == "background"
    assert all(row["class_name"] != "background" for row in quiet.distribution["classes"])

    dataset = ImageClassifierFromDetectBoxDataset(
        [csv_path],
        class_groups=class_groups,
        min_side_px=1,
        background_ratio=1.0,
        background_iou=0.3,
    )
    assert len(dataset) == 2
    stored_path = str(dataset.sample_path_table["img_path"].iloc[0])
    assert dataset._exclusion_boxes[stored_path].shape[0] == 2
    _, positive = dataset[0]
    assert positive["class_name"] == "person"
    _, background = dataset[1]
    assert background["class_id"] == 1
    assert background["class_name"] == "background"
    random.seed(0)
    for _ in range(10):
        path, cx, cy, width, height = dataset._sample_background_box()
        zones = dataset._exclusion_boxes[path]
        overlap = max_iou_cxcywh(cx, cy, width * dataset.box_scale, height * dataset.box_scale, zones)
        assert overlap < 0.3


def test_background_requires_the_name_and_rejects_a_covered_image(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    with pytest.raises(ValueError, match="background"):
        ImageClassifierFromDetectBoxDataset(
            [csv_path],
            class_groups={0: [0]},
            min_side_px=1,
            background_ratio=0.5,
        )
    with pytest.raises(ValueError, match="多个"):
        ImageClassifierFromDetectBoxDataset(
            [csv_path],
            class_groups={0: "background", 1: "background"},
            min_side_px=1,
        )

    img_path = tmp_path / "full.jpg"
    cv2.imwrite(str(img_path), np.zeros((64, 64, 3), dtype=np.uint8))
    covered = tmp_path / "covered.csv"
    _write_box_csv(covered, [_box_row(img_path, "person", 0, 0.5, 0.5, 1.0, 1.0)])
    covered_dataset = ImageClassifierFromDetectBoxDataset(
        [covered],
        class_groups={0: [0], 1: "background"},
        min_side_px=1,
        background_ratio=1.0,
    )
    with pytest.raises(RuntimeError, match="禁区"):
        covered_dataset[1]


def test_class_groups_keep_background_id_outside_the_groups(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    dataset = ImageClassifierFromDetectBoxDataset(
        [csv_path],
        class_groups={0: "background", 1: [0, 1, 3], 2: [2]},
        min_side_px=1,
        background_ratio=1.0,
    )
    assert dataset.map_class_id_to_class_name[0] == "background"
    assert 0 not in dataset.class_groups
    assert len(dataset) == 8
    assert all(row["class_name"] != "background" for row in dataset.distribution["classes"])
    with pytest.raises(ValueError, match="连续"):
        ImageClassifierFromDetectBoxDataset(
            [csv_path],
            class_groups={0: "background", 2: [2]},
            min_side_px=1,
        )


def test_datamodule_applies_background_ratio_only_to_train(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write_age_csv(csv_path)
    module = BoxCropClassifierDataModule(
        train_csv_paths=[str(csv_path)],
        val_csv_paths=[str(csv_path)],
        test_csv_paths=[str(csv_path)],
        predict_csv_paths=[str(csv_path)],
        transform_train=None,
        transform_val=None,
        class_groups={0: [0, 1, 3], 1: [2], 2: "background"},
        min_side_px=1,
        background_ratio=1.0,
        batch_size=2,
        num_workers=0,
    )
    module.setup(None)
    assert len(module.train_dataset) == 8
    assert len(module.val_dataset) == 4
    assert len(module.test_dataset) == 4
    assert len(module.pred_dataset) == 4
    assert module.train_dataset.map_class_id_to_class_name[2] == "background"
