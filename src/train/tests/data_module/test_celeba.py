"""用小型属性表验证 CelebA 的划分、取值和 CSV。不读真实的 20 万张图。"""

from pathlib import Path

import pandas as pd
import pytest

from lovely_deep_learning.data_module import celeba


def _write_tables(root: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """``rows`` 是 ``(文件名, 划分, 身份, 第一个属性的 1 或 -1)``。"""
    (root / "Anno").mkdir(parents=True)
    (root / "Eval").mkdir()
    names = [f"Attr{i:02d}" for i in range(celeba.CELEBA_ATTR_COUNT)]
    attr_lines = [str(len(rows)), " ".join(names)]
    partition_lines = []
    identity_lines = []
    for filename, split, person, mark in rows:
        values = " ".join([mark] + ["-1"] * (celeba.CELEBA_ATTR_COUNT - 1))
        attr_lines.append(f"{filename} {values}")
        partition_lines.append(f"{filename} {split}")
        identity_lines.append(f"{filename} {person}")
    (root / celeba.CELEBA_ATTR_FILE).write_text("\n".join(attr_lines) + "\n", encoding="utf-8")
    (root / celeba.CELEBA_PARTITION_FILE).write_text(
        "\n".join(partition_lines) + "\n", encoding="utf-8"
    )
    (root / celeba.CELEBA_IDENTITY_FILE).write_text(
        "\n".join(identity_lines) + "\n", encoding="utf-8"
    )
    for filename, _, _, _ in rows:
        image = root / celeba.CELEBA_IMAGE_DIR / filename
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")


def _use_tiny_counts(monkeypatch, rows: int, identities: int, counts: dict[str, int]) -> None:
    monkeypatch.setattr(celeba, "CELEBA_IMAGE_COUNT", rows)
    monkeypatch.setattr(celeba, "CELEBA_IDENTITY_COUNT", identities)
    monkeypatch.setattr(celeba, "CELEBA_SPLIT_COUNTS", counts)
    monkeypatch.setattr(celeba, "jpeg_size", lambda path: celeba.CELEBA_IMAGE_SIZE)


def test_attribute_values_map_negative_to_zero(tmp_path: Path):
    path = tmp_path / "list_attr_celeba.txt"
    names = " ".join(f"Attr{i:02d}" for i in range(celeba.CELEBA_ATTR_COUNT))
    values = " ".join(["1"] + ["-1"] * (celeba.CELEBA_ATTR_COUNT - 1))
    path.write_text(f"1\n{names}\n000001.jpg {values}\n", encoding="utf-8")

    parsed, filenames, label = celeba.load_attribute_table(path)

    assert parsed[0] == "Attr00"
    assert filenames == ["000001.jpg"]
    assert label[0, 0] == 1
    assert label[1, 0] == 0
    assert celeba.image_relpath("000001.jpg") == "Img/img_align_celeba/000001.jpg"


def test_generate_csvs_uses_official_split_and_keeps_identities_apart(tmp_path: Path, monkeypatch):
    rows = [
        ("000001.jpg", "0", "10", "1"),
        ("000002.jpg", "0", "11", "-1"),
        ("000003.jpg", "1", "12", "1"),
        ("000004.jpg", "2", "13", "-1"),
    ]
    _write_tables(tmp_path, rows)
    _use_tiny_counts(monkeypatch, 4, 4, {"train": 2, "val": 1, "test": 1})

    celeba.generate_celeba_csvs(tmp_path)

    train = pd.read_csv(tmp_path / "train.csv")
    val = pd.read_csv(tmp_path / "val.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert list(train.columns[:7]) == ["path_img", "cx", "cy", "w", "h", "img_w", "img_h"]
    assert list(train.columns[7:9]) == ["attr_Attr00", "attr_Attr01"]
    assert list(train["path_img"]) == [
        "Img/img_align_celeba/000001.jpg",
        "Img/img_align_celeba/000002.jpg",
    ]
    assert list(val["path_img"]) == ["Img/img_align_celeba/000003.jpg"]
    assert list(test["path_img"]) == ["Img/img_align_celeba/000004.jpg"]
    assert train["attr_Attr00"].tolist() == [1, 0]
    assert int(train.iloc[0]["img_w"]) == 178
    assert int(train.iloc[0]["img_h"]) == 218
    assert set(train["cx"]) == {0.5}
    assert set(train["w"]) == {1}

    (tmp_path / "train.csv").write_text("path_img\n", encoding="utf-8")
    celeba.generate_celeba_csvs(tmp_path)
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "path_img\n"


def test_generate_finds_nested_celeba_directory(tmp_path: Path, monkeypatch):
    nested = tmp_path / "CelebA"
    _write_tables(nested, [("000001.jpg", "0", "7", "1")])
    _use_tiny_counts(monkeypatch, 1, 1, {"train": 1, "val": 0, "test": 0})

    root = celeba.generate_celeba_csvs(tmp_path)

    assert root == nested.resolve()
    assert (nested / "train.csv").is_file()


def test_overlapping_identity_is_rejected(tmp_path: Path, monkeypatch):
    _write_tables(
        tmp_path,
        [
            ("000001.jpg", "0", "7", "1"),
            ("000002.jpg", "1", "7", "-1"),
        ],
    )
    _use_tiny_counts(monkeypatch, 2, 1, {"train": 1, "val": 1, "test": 0})

    with pytest.raises(RuntimeError, match="相同身份"):
        celeba.generate_celeba_csvs(tmp_path)


def test_missing_annotations_name_the_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="list_attr_celeba.txt"):
        celeba.generate_celeba_csvs(tmp_path / "empty")


def test_jpeg_size_must_match_aligned_face(tmp_path: Path, monkeypatch):
    _write_tables(tmp_path, [("000001.jpg", "0", "7", "1")])
    _use_tiny_counts(monkeypatch, 1, 1, {"train": 1, "val": 0, "test": 0})
    monkeypatch.setattr(celeba, "jpeg_size", lambda path: (10, 10))

    with pytest.raises(RuntimeError, match="178×218"):
        celeba.generate_celeba_csvs(tmp_path)

    assert not (tmp_path / "train.csv").exists()
