"""用小型 YOLO 标签树验证 ``generate_coco_csvs``，以及 COCO 2017 的下载与解压。"""

import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from lovely_deep_learning.data_module.coco import (
    COCODataModule,
    _LiveLine,
    _download_zip,
    ensure_coco_files,
    generate_coco_csvs,
)


def _write_labels(root: Path) -> None:
    train = root / "labels" / "train2017"
    val = root / "labels" / "val2017"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "000000000030.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (train / "000000000009.txt").write_text("2 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    (val / "000000000139.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    (root / "train2017.txt").write_text(
        "./images/train2017/000000000009.jpg\n"
        "./images/train2017/000000000030.jpg\n",
        encoding="utf-8",
    )
    (root / "val2017.txt").write_text(
        "./images/val2017/000000000139.jpg\n",
        encoding="utf-8",
    )
    (root / "test-dev2017.txt").write_text(
        "./images/test2017/000000000001.jpg\n",
        encoding="utf-8",
    )


def test_generate_csv_from_directory_and_symlink(tmp_path: Path):
    direct = tmp_path / "direct"
    _write_labels(direct)
    generate_coco_csvs(direct)

    train = pd.read_csv(direct / "train.csv")
    assert list(train["path_img"]) == [
        "images/train2017/000000000009.jpg",
        "images/train2017/000000000030.jpg",
    ]
    assert list(train["path_label_detect_yolo"]) == [
        "labels/train2017/000000000009.txt",
        "labels/train2017/000000000030.txt",
    ]

    val = pd.read_csv(direct / "val.csv")
    assert val.iloc[0]["path_img"] == "images/val2017/000000000139.jpg"
    assert val.iloc[0]["path_label_detect_yolo"] == "labels/val2017/000000000139.txt"

    test = pd.read_csv(direct / "test.csv")
    assert list(test.columns) == ["path_img"]
    assert list(test["path_img"]) == ["images/test2017/000000000001.jpg"]

    predict = pd.read_csv(direct / "predict.csv")
    assert list(predict.columns) == ["path_img"]
    assert list(predict["path_img"]) == list(val["path_img"])

    class_map = pd.read_csv(direct / "map_class_id_to_class_name.csv")
    assert len(class_map) == 80
    assert class_map.iloc[0]["class_name"] == "person"
    assert int(class_map.iloc[0]["class_id"]) == 0

    linked = tmp_path / "linked"
    linked.symlink_to(direct, target_is_directory=True)
    (direct / "train.csv").unlink()
    (direct / "val.csv").unlink()
    (direct / "test.csv").unlink()
    (direct / "predict.csv").unlink()
    (direct / "map_class_id_to_class_name.csv").unlink()
    generate_coco_csvs(linked)
    linked_train = pd.read_csv(linked / "train.csv")
    assert list(linked_train["path_img"]) == list(train["path_img"])
    assert list(linked_train["path_label_detect_yolo"]) == list(
        train["path_label_detect_yolo"]
    )
    assert not str(linked_train.iloc[0]["path_img"]).startswith("/")


def test_skip_existing_csv_unless_overwrite(tmp_path: Path):
    root = tmp_path / "coco"
    _write_labels(root)
    generate_coco_csvs(root)
    (root / "train.csv").write_text(
        "path_img,path_label_detect_yolo\n", encoding="utf-8"
    )

    generate_coco_csvs(root)
    assert pd.read_csv(root / "train.csv").empty

    generate_coco_csvs(root, overwrite=True)
    assert len(pd.read_csv(root / "train.csv")) == 2


def test_missing_labels_raises(tmp_path: Path):
    root = tmp_path / "coco"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="image list not found"):
        generate_coco_csvs(root)


def test_broken_symlink_raises(tmp_path: Path):
    root = tmp_path / "coco"
    root.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="dataset_dir not found"):
        generate_coco_csvs(root)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _write_fixture_zips(root: Path) -> None:
    _write_zip(
        root / "coco2017labels.zip",
        {
            "coco/labels/train2017/000000000009.txt": b"0 0.5 0.5 0.1 0.1\n",
            "coco/labels/val2017/000000000139.txt": b"0 0.4 0.4 0.1 0.1\n",
            "coco/README.txt": b"readme\n",
            "coco/train2017.txt": (
                b"./images/train2017/000000000009.jpg\n"
                b"./images/train2017/000000000025.jpg\n"
            ),
            "coco/val2017.txt": b"./images/val2017/000000000139.jpg\n",
            "coco/test-dev2017.txt": b"./images/test2017/000000000001.jpg\n",
            "coco/images/train2017/": b"",
        },
    )
    _write_zip(
        root / "train2017.zip",
        {
            "train2017/000000000009.jpg": b"train-a",
            "train2017/000000000025.jpg": b"train-b",
        },
    )
    _write_zip(root / "val2017.zip", {"val2017/000000000139.jpg": b"val"})
    _write_zip(root / "test2017.zip", {"test2017/000000000001.jpg": b"test"})


def test_ensure_extracts_labels_without_coco_prefix_and_images(tmp_path: Path):
    root = tmp_path / "coco"
    _write_fixture_zips(root)
    ensure_coco_files(root)

    assert (root / "labels" / "train2017" / "000000000009.txt").read_bytes().startswith(b"0 ")
    assert (root / "labels" / "val2017" / "000000000139.txt").is_file()
    assert (root / "README.txt").read_text(encoding="utf-8") == "readme\n"
    assert (root / "train2017.txt").is_file()
    assert (root / "test-dev2017.txt").is_file()
    assert not (root / "coco").exists()
    assert (root / "images" / "train2017" / "000000000009.jpg").read_bytes() == b"train-a"
    assert (root / "images" / "train2017" / "000000000025.jpg").read_bytes() == b"train-b"
    assert (root / "images" / "val2017" / "000000000139.jpg").read_bytes() == b"val"
    assert (root / "images" / "test2017" / "000000000001.jpg").read_bytes() == b"test"
    assert (root / "train2017.txt").read_text(encoding="utf-8").count("images/train2017/") == 2


def test_ensure_leaves_a_split_that_already_has_a_file(tmp_path, monkeypatch):
    root = tmp_path / "coco"
    _write_fixture_zips(root)
    train_dir = root / "images" / "train2017"
    train_dir.mkdir(parents=True)
    (train_dir / "000000000009.jpg").write_bytes(b"keep")

    def fail_download(url, dest):
        raise AssertionError(f"unexpected download {url} -> {dest}")

    monkeypatch.setattr(
        "lovely_deep_learning.data_module.coco._download_zip", fail_download
    )
    ensure_coco_files(root)
    assert (train_dir / "000000000009.jpg").read_bytes() == b"keep"
    assert not (train_dir / "000000000025.jpg").exists()
    assert (root / "README.txt").is_file()
    assert (root / "images" / "val2017" / "000000000139.jpg").read_bytes() == b"val"


def test_prepare_data_writes_csv_after_extract(tmp_path: Path):
    root = tmp_path / "coco"
    _write_fixture_zips(root)
    module = COCODataModule(
        train_csv_paths=[str(root / "train.csv")],
        val_csv_paths=[str(root / "val.csv")],
        test_csv_paths=[str(root / "val.csv")],
        predict_csv_paths=[str(root / "predict.csv")],
        transform_train=None,
        transform_val=None,
        dataset_dir=str(root),
    )
    module.prepare_data()
    train = pd.read_csv(root / "train.csv")
    assert list(train["path_img"]) == [
        "images/train2017/000000000009.jpg",
        "images/train2017/000000000025.jpg",
    ]
    assert list(train["path_label_detect_yolo"]) == [
        "labels/train2017/000000000009.txt",
        "labels/train2017/000000000025.txt",
    ]
    assert not (root / "labels" / "train2017" / "000000000025.txt").exists()
    test = pd.read_csv(root / "test.csv")
    assert list(test.columns) == ["path_img"]
    assert list(test["path_img"]) == ["images/test2017/000000000001.jpg"]
    assert (root / "map_class_id_to_class_name.csv").is_file()


def test_prepare_data_is_silent_when_splits_exist(tmp_path, monkeypatch, capsys):
    root = tmp_path / "coco"
    for relative in (
        "labels/train2017",
        "labels/val2017",
        "images/train2017",
        "images/val2017",
        "images/test2017",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "000000000001.txt").write_text("x\n", encoding="utf-8")
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    predict_csv = root / "predict.csv"
    train_csv.write_text(
        "path_img,path_label_detect_yolo\na.jpg,a.txt\nb.jpg,b.txt\n",
        encoding="utf-8",
    )
    val_csv.write_text(
        "path_img,path_label_detect_yolo\nv.jpg,v.txt\n",
        encoding="utf-8",
    )
    predict_csv.write_text("path_img\nv.jpg\n", encoding="utf-8")

    def fail_download(url, dest):
        raise AssertionError(f"unexpected download {url} -> {dest}")

    monkeypatch.setattr(
        "lovely_deep_learning.data_module.coco._download_zip", fail_download
    )
    module = COCODataModule(
        train_csv_paths=[str(train_csv)],
        val_csv_paths=[str(val_csv)],
        test_csv_paths=[str(val_csv)],
        predict_csv_paths=[str(predict_csv)],
        transform_train=None,
        transform_val=None,
        dataset_dir=str(root),
    )
    module.prepare_data()
    assert capsys.readouterr().out == ""
    assert train_csv.read_text(encoding="utf-8").count("a.jpg") == 1


def test_prepare_data_prints_root_once(tmp_path: Path, capsys):
    root = tmp_path / "coco"
    _write_fixture_zips(root)
    module = COCODataModule(
        train_csv_paths=[str(root / "train.csv")],
        val_csv_paths=[str(root / "val.csv")],
        test_csv_paths=[str(root / "val.csv")],
        predict_csv_paths=[str(root / "predict.csv")],
        transform_train=None,
        transform_val=None,
        dataset_dir=str(root),
    )
    module.prepare_data()
    output = capsys.readouterr().out
    assert "=" * 70 in output
    assert output.count(f"COCO  {root.as_posix()}") == 1
    assert "->" not in output
    assert "压缩包" in output
    assert "coco2017labels.zip" in output
    assert "已有" in output
    assert "解压，去掉 coco/ 前缀" in output
    assert "images/train2017" in output
    assert "解压中" in output
    assert "个文件" in output
    assert "\nCSV\n" in output
    assert "train.csv" in output
    assert "写入" in output


def test_generate_prints_symlink_target(tmp_path: Path, capsys):
    direct = tmp_path / "direct"
    _write_labels(direct)
    linked = tmp_path / "linked"
    linked.symlink_to(direct, target_is_directory=True)
    generate_coco_csvs(linked)
    output = capsys.readouterr().out
    assert f"COCO  {linked.as_posix()} -> {direct.resolve().as_posix()}" in output
    assert "\nCSV\n" in output
    assert "写入" in output


def test_progress_lines_every_tenth_when_not_a_tty(capsys):
    line = _LiveLine()
    assert line._tty is False
    for pct in range(1, 101):
        line.show(f"zip {pct}", percent=pct)
    line.show("zip done", final=True)
    printed = [text for text in capsys.readouterr().out.splitlines() if text]
    assert printed == [f"zip {pct}" for pct in range(10, 100, 10)] + ["zip done"]


def test_missing_label_file_is_background(tmp_path: Path):
    from lovely_deep_learning.dataset.object_detect import ObjectDetectDataset

    cls, bbox = ObjectDetectDataset.read_yolo_detection_labels(
        str(tmp_path / "missing.txt")
    )
    assert cls.shape == (0,)
    assert bbox.shape == (0, 4)

    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    cls, bbox = ObjectDetectDataset.read_yolo_detection_labels(str(empty))
    assert cls.shape == (0,)
    assert bbox.shape == (0, 4)


def test_download_resumes_partial_file(tmp_path, monkeypatch):
    dest = tmp_path / "train2017.zip"
    dest.write_bytes(b"PK")

    class _Body:
        status = 206

        def __init__(self):
            self._data = b"MORE"

        def read(self, n=-1):
            if n is None or n < 0 or n >= len(self._data):
                data, self._data = self._data, b""
                return data
            data, self._data = self._data[:n], self._data[n:]
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, *args, **kwargs):
        assert request.get_header("Range") == "bytes=2-"
        return _Body()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    _download_zip("http://example.invalid/train2017.zip", dest)
    assert dest.read_bytes() == b"PKMORE"


def test_download_restarts_when_range_is_rejected(tmp_path, monkeypatch):
    dest = tmp_path / "val2017.zip"
    dest.write_bytes(b"partial")
    calls = {"n": 0}

    class _Body:
        status = 200

        def __init__(self, data):
            self._data = data

        def read(self, n=-1):
            if n is None or n < 0 or n >= len(self._data):
                data, self._data = self._data, b""
                return data
            data, self._data = self._data[:n], self._data[n:]
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 416, "Range Not Satisfiable", hdrs=None, fp=None
            )
        return _Body(b"FULL")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    _download_zip("http://example.invalid/val2017.zip", dest)
    assert dest.read_bytes() == b"FULL"
    assert calls["n"] == 2


def test_reject_label_zip_path_escape(tmp_path: Path):
    root = tmp_path / "coco"
    root.mkdir()
    _write_zip(
        root / "coco2017labels.zip",
        {
            "coco/labels/train2017/000000000009.txt": b"0 0.5 0.5 0.1 0.1\n",
            "coco/labels/val2017/000000000139.txt": b"0 0.4 0.4 0.1 0.1\n",
            "coco/../../outside.txt": b"nope\n",
        },
    )
    _write_zip(root / "train2017.zip", {"train2017/000000000009.jpg": b"a"})
    _write_zip(root / "val2017.zip", {"val2017/000000000139.jpg": b"b"})
    _write_zip(root / "test2017.zip", {"test2017/000000000001.jpg": b"c"})
    with pytest.raises(RuntimeError, match="非法压缩包路径"):
        ensure_coco_files(root)
    assert not (tmp_path / "outside.txt").exists()
