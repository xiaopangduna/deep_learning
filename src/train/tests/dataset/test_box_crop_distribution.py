from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lovely_deep_learning.data_module.box_crop_classifier import BoxCropClassifierDataModule
from lovely_deep_learning.dataset.box_crop_classifier import BoxCropClassifierDataset
from lovely_deep_learning.dataset.box_crop_distribution import (
    _short_side_bins,
    format_distribution_report,
    training_class_rows,
)

PERSON_CAR = {0: "person", 1: "car"}


def _row(path, class_name, class_id, cx, cy, w, h, img_w=100, img_h=100):
    return {
        "path_img": path,
        "class_name": class_name,
        "class_id": class_id,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "img_w": img_w,
        "img_h": img_h,
    }


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _train_rows():
    return [
        _row("img_a.jpg", "person", 0, 0.5, 0.5, 0.5, 0.4),
        _row("img_a.jpg", "car", 2, 0.5, 0.5, 0.2, 0.2),
        _row("img_z.jpg", "person", 0, 0.5, 0.5, 0, 0.5),
        _row("img_z.jpg", "person", 0, 0.5, 0.5, 0.1, 0.1),
        _row("img_z.jpg", "dog", 18, 0.5, 0.5, 0.5, 0.5),
        _row("img_b.jpg", "car", 2, 0.05, 0.5, 0.2, 0.2),
        _row("img_z.jpg", "dog", 18, 0.5, 0.5, -0.1, 0.2),
        _row("img_z.jpg", "dog", 18, 0.5, 0.5, 0.05, 0.05),
    ]


def test_distribution_funnel_is_exclusive_and_geometry_matches(tmp_path, capsys):
    csv_path = tmp_path / "train.csv"
    _write(csv_path, _train_rows())
    dataset = BoxCropClassifierDataset(
        [csv_path],
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=16,
        box_scale=1.2,
    )
    assert "map_class_id_to_class_name 为空" not in capsys.readouterr().out
    report = dataset.distribution
    assert report["n_raw"] == 8
    assert report["dropped"] == {"invalid_wh": 2, "short_side": 2, "class_excluded": 1}
    assert report["n_kept"] == 3
    assert report["n_images"] == 2
    assert report["boxes_per_image"] == pytest.approx(1.5)
    by_id = {item["class_id"]: item for item in report["classes"]}
    assert by_id[0]["count"] == 1
    assert by_id[1]["count"] == 2
    assert by_id[0]["weight"] == pytest.approx(1.5)
    assert by_id[1]["weight"] == pytest.approx(0.75)
    assert report["suggested_weight_class_ids"] == [0, 1]
    geometry = report["geometry"]
    assert geometry["short_side_px"]["p5"] == pytest.approx(20)
    assert geometry["short_side_px"]["p50"] == pytest.approx(20)
    assert geometry["short_side_px"]["p95"] == pytest.approx(38)
    assert geometry["short_side_bins"] == {
        "lt_16": 0,
        "16_32": 2,
        "32_64": 1,
        "64_128": 0,
        "128_224": 0,
        "ge_224": 0,
    }
    assert geometry["aspect"]["p95"] == pytest.approx(1.225)
    assert geometry["area_px"]["p95"] == pytest.approx(1840)
    assert geometry["clip_ratio"] == pytest.approx(1 / 3)
    assert report["mapping"]["consistent"] is True
    assert report["mapping"]["span"] == "0-1"
    root = csv_path.parent.resolve()
    assert dataset.kept_image_paths == {str(root / "img_a.jpg"), str(root / "img_b.jpg")}


def test_short_side_bin_edges():
    bins = _short_side_bins(np.array([15.9, 16, 31.9, 32, 223.9, 224]))
    assert bins["lt_16"] == 1
    assert bins["16_32"] == 2
    assert bins["32_64"] == 1
    assert bins["128_224"] == 1
    assert bins["ge_224"] == 1
    assert bins["64_128"] == 0


def test_missing_class_weight_is_not_a_short_vector(tmp_path):
    csv_path = tmp_path / "train.csv"
    _write(
        csv_path,
        [
            _row("img_a.jpg", "person", 0, 0.5, 0.5, 0.4, 0.4),
            _row("img_b.jpg", "car", 2, 0.5, 0.5, 0.1, 0.1),
        ],
    )
    dataset = BoxCropClassifierDataset(
        [csv_path],
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=16,
    )
    by_id = {item["class_id"]: item for item in dataset.distribution["classes"]}
    assert by_id[1]["count"] == 0
    assert by_id[1]["weight"] is None
    assert dataset.distribution["suggested_weight_class_ids"] == [0]
    text = format_distribution_report(
        {
            "splits": [
                {
                    "title": "训练集",
                    "csv_paths": [str(csv_path)],
                    "classes": training_class_rows(dataset.distribution["classes"]),
                }
            ]
        }
    )
    assert "class_id" in text and "class_name" in text and "count" in text and "ratio" in text
    header = next(line for line in text.splitlines() if line.strip().startswith("class_id"))
    assert header.split() == ["class_id", "class_name", "count", "ratio"]
    assert "car" in text
    assert "0.000" in text
    assert "短边" not in text


def test_class_groups_report_member_counts(tmp_path):
    csv_path = tmp_path / "age.csv"
    _write(
        csv_path,
        [
            _row(f"img_{index}.jpg", name, class_id, 0.5, 0.5, 0.4, 0.4)
            for index, (class_id, name) in enumerate(
                [(0, "head-age_0"), (1, "head-age_1"), (2, "head-age_2"), (3, "head-age_3"), (4, "other")]
            )
        ],
    )
    dataset = BoxCropClassifierDataset(
        [csv_path],
        class_groups={0: [0, 1, 3], 1: [2]},
        min_side_px=16,
    )
    report = dataset.distribution
    assert report["dropped"]["class_excluded"] == 1
    assert report["n_kept"] == 4
    by_id = {item["class_id"]: item for item in report["classes"]}
    assert by_id[0]["count"] == 3
    assert by_id[0]["weight"] == pytest.approx(4 / 6)
    assert [(item["class_id"], item["count"]) for item in by_id[0]["members"]] == [
        (0, 1),
        (1, 1),
        (3, 1),
    ]
    assert by_id[1]["members"] == [
        {"class_id": 2, "class_name": "head-age_2", "count": 1}
    ]
    text = format_distribution_report(
        {
            "splits": [
                {
                    "title": "训练集",
                    "csv_paths": [str(csv_path)],
                    "classes": training_class_rows(report["classes"]),
                }
            ]
        }
    )
    assert "head-age_0+head-age_1+head-age_3" in text
    assert "head-age_2" in text
    assert "原始" not in text


def test_fit_prints_train_and_val_class_tables(tmp_path, capsys):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _write(train_csv, _train_rows())
    _write(
        val_csv,
        [
            _row("img_a.jpg", "person", 0, 0.5, 0.5, 0.5, 0.4),
            _row("img_c.jpg", "car", 2, 0.5, 0.5, 0.3, 0.3),
        ],
    )
    _write(
        test_csv,
        [
            _row("img_c.jpg", "car", 2, 0.5, 0.5, 0.3, 0.3),
            _row("img_d.jpg", "person", 0, 0.5, 0.5, 0.4, 0.4),
        ],
    )
    log_dir = tmp_path / "logs" / "version_0"
    module = BoxCropClassifierDataModule(
        train_csv_paths=[str(train_csv)],
        val_csv_paths=[str(val_csv)],
        test_csv_paths=[str(test_csv)],
        predict_csv_paths=[str(val_csv)],
        transform_train=None,
        transform_val=None,
        map_class_id_to_class_name=PERSON_CAR,
        min_side_px=16,
        box_scale=1.2,
        batch_size=2,
        num_workers=0,
    )
    module.trainer = SimpleNamespace(is_global_zero=True, log_dir=str(log_dir))
    module.setup("fit")
    assert module.test_dataset is None
    text = capsys.readouterr().out
    assert "训练集" in text
    assert "验证集" in text
    assert "测试集" not in text
    assert "过滤前" not in text
    assert "短边" not in text
    assert "weight" not in text
    assert "person" in text and "car" in text
    assert "====" in text and "----" in text
    saved = (log_dir / "data_distribution.txt").read_text(encoding="utf-8")
    assert not (log_dir / "data_distribution.json").exists()
    assert saved == text.split("数据分布已写入", 1)[0]
    train_block = saved.split("验证集", 1)[0]
    assert "person" in train_block and "0.333" in train_block
    assert "car" in train_block and "0.667" in train_block
    assert "数据分布已写入" in text
