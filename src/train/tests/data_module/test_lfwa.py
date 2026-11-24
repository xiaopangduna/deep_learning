"""用小型 tar 验证 LFW 图像包的统计、解压和缺标签说明。不访问网络。"""

import io
import tarfile
from pathlib import Path

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


def test_generate_csvs_maps_path_and_splits(tmp_path: Path):
    root = tmp_path / "lfw"
    root.mkdir()
    attr = root / lfwa.LFW_ATTR_NAME
    attr.write_text(
        _attr_text(
            [
                ("George_HW_Bush_0001.jpg", "1"),
                ("Ada_0001.jpg", "-1"),
                ("Bea_0002.jpg", "1"),
            ]
        ),
        encoding="utf-8",
    )

    lfwa.generate_lfwa_csvs(root, expected_images=3, train_count=2)

    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    assert list(train["path_img"]) == [
        "lfw/George_HW_Bush/George_HW_Bush_0001.jpg",
        "lfw/Ada/Ada_0001.jpg",
    ]
    assert int(train.iloc[0]["Attr00"]) == 1
    assert int(train.iloc[1]["Attr00"]) == 0
    assert list(test["path_img"]) == ["lfw/Bea/Bea_0002.jpg"]

    (root / "train.csv").write_text("path_img\n", encoding="utf-8")
    lfwa.generate_lfwa_csvs(root, expected_images=3, train_count=2)
    assert (root / "train.csv").read_text(encoding="utf-8") == "path_img\n"


def test_missing_attr_file_names_the_manual_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="drive.google.com"):
        lfwa.generate_lfwa_csvs(tmp_path / "empty")


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
