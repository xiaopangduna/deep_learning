"""用小型 ILSVRC 树验证 ``generate_imagenet_csvs``，包含 ``ILSVRC`` 软链接。"""

from pathlib import Path

import pandas as pd
import pytest

from lovely_deep_learning.data_module.image_net import generate_imagenet_csvs


def _write_ilsvrc(ilsvrc: Path) -> None:
    ann = ilsvrc / "Annotations" / "CLS-LOC" / "val"
    ann.mkdir(parents=True)
    (ann / "ILSVRC2012_val_00000001.xml").write_text(
        "<annotation><object><name>n01443537</name></object></annotation>",
        encoding="utf-8",
    )
    sets = ilsvrc / "ImageSets" / "CLS-LOC"
    sets.mkdir(parents=True)
    (sets / "train_cls.txt").write_text(
        "n01440764/n01440764_18 1\n"
        "n09999999/missing 1\n"
        "n01443537/n01443537_1 2\n",
        encoding="utf-8",
    )
    (sets / "val.txt").write_text(
        "ILSVRC2012_val_00000001 1\n"
        "ILSVRC2012_val_00000002 2\n",
        encoding="utf-8",
    )
    (sets / "test.txt").write_text("ILSVRC2012_test_00000001 0\n", encoding="utf-8")


def test_generate_csv_from_directory_and_symlink(tmp_path: Path):
    direct = tmp_path / "direct"
    _write_ilsvrc(direct / "ILSVRC")
    generate_imagenet_csvs(direct)

    train = pd.read_csv(direct / "train.csv")
    assert list(train["class_name"]) == ["n01440764", "n09999999", "n01443537"]
    assert list(train["class_id"]) == [0, 2, 1]
    assert train.iloc[0]["path_img"] == (
        "ILSVRC/Data/CLS-LOC/train/n01440764/n01440764_18.JPEG"
    )

    val = pd.read_csv(direct / "val.csv")
    assert len(val) == 1
    assert val.iloc[0]["class_name"] == "n01443537"
    assert val.iloc[0]["class_id"] == 1

    test = pd.read_csv(direct / "test.csv")
    predict = pd.read_csv(direct / "predict.csv")
    assert list(test["path_img"]) == list(predict["path_img"])
    assert test.iloc[0]["path_img"].endswith("ILSVRC2012_test_00000001.JPEG")

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "ILSVRC").symlink_to(direct / "ILSVRC", target_is_directory=True)
    generate_imagenet_csvs(linked)
    linked_train = pd.read_csv(linked / "train.csv")
    assert list(linked_train["path_img"]) == list(train["path_img"])
    assert not str(linked_train.iloc[0]["path_img"]).startswith("/")


def test_skip_existing_csv_unless_overwrite(tmp_path: Path):
    root = tmp_path / "imagenet"
    _write_ilsvrc(root / "ILSVRC")
    generate_imagenet_csvs(root)
    (root / "train.csv").write_text("path_img,class_name,class_id\n", encoding="utf-8")

    generate_imagenet_csvs(root)
    assert pd.read_csv(root / "train.csv").empty

    generate_imagenet_csvs(root, overwrite=True)
    assert len(pd.read_csv(root / "train.csv")) == 3


def test_broken_symlink_raises(tmp_path: Path):
    root = tmp_path / "imagenet"
    root.mkdir()
    (root / "ILSVRC").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="symlink is broken"):
        generate_imagenet_csvs(root)
