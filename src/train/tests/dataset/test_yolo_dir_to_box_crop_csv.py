import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "yolo_dir_to_box_crop_csv.py"
_spec = importlib.util.spec_from_file_location("yolo_dir_to_box_crop_csv", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _write_dataset(root: Path, real_image: Path) -> None:
    root.mkdir(parents=True)
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"names": {0: "head-age_0", 1: "head-age_1", 2: "head-age_2"}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    images = root / "images" / "train_v011"
    labels = root / "labels" / "train_v011"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").symlink_to(real_image)
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n",
        encoding="utf-8",
    )
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(images / "empty.jpg")
    (labels / "empty.txt").write_text("", encoding="utf-8")
    (images / "unknown.png").symlink_to(real_image)
    (labels / "unknown.txt").write_text("9 0.2 0.3 0.4 0.5\n", encoding="utf-8")


def test_convert_one_csv_resolves_symlink_and_skips_empty(tmp_path: Path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_image = real_dir / "source.jpg"
    Image.new("RGB", (20, 10), color=(4, 5, 6)).save(real_image)
    data_root = tmp_path / "yolo"
    _write_dataset(data_root, real_image)
    out_csv = tmp_path / "out" / "train.csv"

    _mod.convert(data_root, out_csv)

    table = pd.read_csv(out_csv)
    assert list(table.columns) == _mod.LABEL_COLUMNS
    assert len(table) == 3
    assert set(table["class_name"]) == {"head-age_0", "head-age_2", "9"}
    linked = table[table["class_name"] == "head-age_0"].iloc[0]
    assert linked["class_id"] == 0
    assert linked["img_w"] == 20
    assert linked["img_h"] == 10
    assert (out_csv.parent / linked["path_img"]).resolve() == real_image.resolve()
    assert "empty" not in "".join(table["path_img"])


def test_split_required_when_multiple(tmp_path: Path):
    data_root = tmp_path / "yolo"
    (data_root / "images" / "a").mkdir(parents=True)
    (data_root / "images" / "b").mkdir()
    (data_root / "dataset.yaml").write_text("names:\n  0: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--split"):
        _mod.resolve_split(data_root, None)
