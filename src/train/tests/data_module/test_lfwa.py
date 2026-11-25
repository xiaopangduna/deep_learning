"""用小型 tar 验证 LFW 图像包的统计、解压和缺标签说明。不访问网络。"""

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lovely_deep_learning.data_module import lfwa


def _write_tgz(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_inspect_counts_images_identities_and_labels(tmp_path: Path):
    archive = tmp_path / "lfw.tgz"
    _write_tgz(
        archive,
        [
            ("lfw/Ada/Ada_0001.jpg", b"a"),
            ("lfw/Ada/Ada_0002.jpg", b"b"),
            ("lfw/Bea/Bea_0001.jpg", b"c"),
            ("lfw/list_attr_lfwa.txt", b"attrs"),
        ],
    )

    info = lfwa.inspect_lfw_tgz(archive)

    assert info.images == 3
    assert info.identities == 2
    assert info.label_names == ("lfw/list_attr_lfwa.txt",)


def test_extract_rejects_parent_path(tmp_path: Path):
    archive = tmp_path / "lfw.tgz"
    _write_tgz(archive, [("../outside.jpg", b"nope")])

    with pytest.raises(RuntimeError, match="非法"):
        lfwa.extract_lfw_tgz(archive, tmp_path / "out")

    assert not (tmp_path / "outside.jpg").exists()


def test_prepare_extracts_once_and_reports_missing_labels(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "data"
    root.mkdir()
    _write_tgz(root / lfwa.LFW_ARCHIVE_NAME, [("lfw/Ada/Ada_0001.jpg", b"jpeg")])
    monkeypatch.setattr(lfwa, "LFW_IMAGE_COUNT", 1)
    monkeypatch.setattr(lfwa, "LFW_IDENTITY_COUNT", 1)
    monkeypatch.setattr(lfwa, "download_lfw_tgz", lambda *args, **kwargs: None)

    info = lfwa.prepare_lfw_images(root)
    image = root / "lfw" / "Ada" / "Ada_0001.jpg"
    assert image.read_bytes() == b"jpeg"
    assert info.label_names == ()
    assert "没有属性标签" in capsys.readouterr().out

    image.write_bytes(b"kept")
    lfwa.prepare_lfw_images(root)
    assert image.read_bytes() == b"kept"


def _attr_text(rows: list[tuple[str, str]]) -> str:
    names = [f"Attr{i:02d}" for i in range(lfwa.LFW_ATTR_COUNT)]
    lines = [str(len(rows)), " ".join(names)]
    for filename, mark in rows:
        values = " ".join([mark] + ["-1"] * (lfwa.LFW_ATTR_COUNT - 1))
        lines.append(f"{filename} {values}")
    return "\n".join(lines) + "\n"


def test_parse_list_attr_maps_values(tmp_path: Path):
    path = tmp_path / lfwa.LFW_ATTR_NAME
    path.write_text(
        _attr_text(
            [
                ("George_HW_Bush_0001.jpg", "1"),
                ("Ada_0001.jpg", "-1"),
            ]
        ),
        encoding="utf-8",
    )

    names, rows = lfwa.parse_list_attr_lfwa(path, expected_images=2, train_count=1)

    assert names[0] == "Attr00"
    assert rows[0][1][0] == 1
    assert rows[1][1][0] == 0
    assert lfwa.image_relpath(rows[0][0]) == "lfw/George_HW_Bush/George_HW_Bush_0001.jpg"


def test_generate_csvs_splits_attribute_sets_and_keeps_whole_identities(tmp_path: Path, monkeypatch):
    root = tmp_path / "lfw"
    paths = [
        "lfw/Ada/Ada_0001.jpg",
        "lfw/Ada/Ada_0002.jpg",
        "lfw/Bea/Bea_0001.jpg",
        "lfw/Cara/Cara_0001.jpg",
    ]
    for relative in paths:
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"")
    label40 = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int64)
    label73 = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int64)

    def fake_attr(path: Path, attr_count: int):
        if attr_count == lfwa.LFW_ATTR_COUNT:
            return ["Young", "Male"], paths, label40
        return ["Eyes Open", "Male"], paths, label73

    monkeypatch.setattr(lfwa, "load_attribute_mat", fake_attr)
    monkeypatch.setattr(
        lfwa,
        "load_split_mat",
        lambda path, image_count: (np.array([0, 1, 2]), np.array([3])),
    )
    monkeypatch.setattr(lfwa, "jpeg_size", lambda path: (250, 250))

    lfwa.generate_lfwa_csvs(root, val_image_target=1, val_seed=0)

    train40 = pd.read_csv(root / "train_40.csv")
    val40 = pd.read_csv(root / "val_40.csv")
    test40 = pd.read_csv(root / "test_40.csv")
    train73 = pd.read_csv(root / "train_73.csv")
    assert list(train40.columns[:7]) == ["path_img", "cx", "cy", "w", "h", "img_w", "img_h"]
    assert list(train40.columns[7:]) == ["attr_Young", "attr_Male"]
    assert list(train73.columns[7:]) == ["attr_Eyes_Open", "attr_Male"]
    assert "attr_Eyes_Open" not in train40.columns
    assert "attr_Young" not in train73.columns
    assert list(train40["path_img"]) == list(train73["path_img"])
    assert list(test40["path_img"]) == ["lfw/Cara/Cara_0001.jpg"]
    assert set(train40["cx"]) == {0.5}
    assert set(train40["w"]) == {1}
    assert int(train40.iloc[0]["img_w"]) == 250
    train_people = {path.split("/")[1] for path in train40["path_img"]}
    val_people = {path.split("/")[1] for path in val40["path_img"]}
    assert train_people.isdisjoint(val_people)
    assert len(val40) >= 1
    assert len(train40) + len(val40) == 3

    (root / "train_40.csv").write_text("path_img\n", encoding="utf-8")
    lfwa.generate_lfwa_csvs(root, val_image_target=1, val_seed=0)
    assert (root / "train_40.csv").read_text(encoding="utf-8") == "path_img\n"


def test_missing_mat_names_the_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="lfw_att_40.mat"):
        lfwa.generate_lfwa_csvs(tmp_path / "empty")


def test_jpeg_size_reads_sof(tmp_path: Path):
    path = tmp_path / "face.jpg"
    path.write_bytes(
        b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\xfa\x00\xfa\x01\x01\x11\x00\xff\xd9"
    )

    assert lfwa.jpeg_size(path) == (250, 250)


def test_attribute_column_and_mat_path():
    assert lfwa.attribute_column("Eyes Open") == "attr_Eyes_Open"
    assert lfwa.mat_image_relpath(r"Ada\Ada_0001.jpg") == "lfw/Ada/Ada_0001.jpg"


def test_download_rejects_html_confirm_page(tmp_path: Path, monkeypatch):
    class _Html:
        status = 200

        def __init__(self):
            self._chunks = [b"<!DOCTYPE html>", b""]

        def read(self, _size):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(lfwa.urllib.request, "urlopen", lambda *args, **kwargs: _Html())

    with pytest.raises(RuntimeError, match="gzip"):
        lfwa.download_lfw_tgz(tmp_path / "lfw.tgz", url="http://example.invalid/lfw.tgz")

    assert not (tmp_path / "lfw.tgz").exists()
