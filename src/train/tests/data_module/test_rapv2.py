"""用小型标注验证 RAP v2 的划分、不确定性别和 CSV。不读真实的 8 万张图。"""

import pickle
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lovely_deep_learning.data_module import rapv2


def _png(path: Path, width: int, height: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _annotation(root: Path, gender: list[int]) -> None:
    rapv2._install_easydict()
    easy = sys.modules["easydict"].EasyDict
    names = [f"Attr{i:02d}" for i in range(rapv2.RAPV2_ATTR_COUNT - 1)] + [rapv2.RAPV2_GENDER_ATTR]
    expand = [f"Color {i}" for i in range(rapv2.RAPV2_EXPAND_COUNT)]
    images = [f"cam{i}.png" for i in range(len(gender))]
    label = np.zeros((len(gender), rapv2.RAPV2_ATTR_COUNT), dtype=np.int32)
    label[:, -1] = gender
    expand_label = np.zeros((len(gender), rapv2.RAPV2_EXPAND_COUNT), dtype=np.int32)
    expand_label[:, 0] = 1
    partition = {
        "train": [np.array([0], dtype=np.int32)],
        "val": [np.array([1], dtype=np.int32)],
        "test": [np.array([2], dtype=np.int32)],
    }
    payload = easy(
        image_name=images,
        attr_name=names,
        expand_attributes=expand,
        label=label,
        expand_label=expand_label,
        partition=partition,
    )
    with (root / rapv2.RAPV2_ANNOTATION_NAME).open("wb") as handle:
        pickle.dump(payload, handle)
    for index, name in enumerate(images):
        _png(root / rapv2.RAPV2_IMAGE_DIR / name, 8 + index, 6)


def _use_tiny_counts(monkeypatch) -> None:
    monkeypatch.setattr(rapv2, "RAPV2_IMAGE_COUNT", 3)
    monkeypatch.setattr(rapv2, "RAPV2_SPLIT_COUNT", 1)
    monkeypatch.setattr(rapv2, "RAPV2_SPLIT_COUNTS", {"train": 1, "val": 1, "test": 1})


def test_png_size_reads_ihdr(tmp_path: Path):
    path = tmp_path / "person.png"
    _png(path, 12, 20)

    assert rapv2.png_size(path) == (12, 20)
    assert rapv2.image_relpath("cam0.png") == "RAP_dataset/cam0.png"


def test_generate_csvs_maps_unknown_gender_and_writes_both_tables(tmp_path: Path, monkeypatch):
    _use_tiny_counts(monkeypatch)
    _annotation(tmp_path, [1, 2, 0])

    rapv2.generate_rapv2_csvs(tmp_path)

    train = pd.read_csv(tmp_path / "train.csv")
    val = pd.read_csv(tmp_path / "val.csv")
    assert not (tmp_path / "test_expand.csv").exists()
    assert train.loc[0, "path_img"] == "RAP_dataset/cam0.png"
    assert train.loc[0, "cx"] == 0.5
    assert train.loc[0, "img_w"] == 8
    assert train.loc[0, f"attr_{rapv2.RAPV2_GENDER_ATTR}"] == 1
    assert train.loc[0, "attr_Color_0"] == 1
    assert val.loc[0, f"attr_{rapv2.RAPV2_GENDER_ATTR}"] == 0
    assert val.loc[0, "img_w"] == 9
    assert len(train.columns) == 7 + rapv2.RAPV2_ATTR_COUNT + rapv2.RAPV2_EXPAND_COUNT
    gender_at = list(train.columns).index(f"attr_{rapv2.RAPV2_GENDER_ATTR}")
    color_at = list(train.columns).index("attr_Color_0")
    assert gender_at < color_at


def test_generate_rejects_unknown_on_other_attributes(tmp_path: Path, monkeypatch):
    _use_tiny_counts(monkeypatch)
    _annotation(tmp_path, [0, 0, 0])
    payload_path = tmp_path / rapv2.RAPV2_ANNOTATION_NAME
    rapv2._install_easydict()
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["label"][0, 0] = 2
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(RuntimeError, match="Attr00"):
        rapv2.generate_rapv2_csvs(tmp_path)


def test_generate_requires_extracted_png(tmp_path: Path, monkeypatch):
    _use_tiny_counts(monkeypatch)
    _annotation(tmp_path, [0, 1, 0])
    (tmp_path / rapv2.RAPV2_IMAGE_DIR / "cam0.png").unlink()

    with pytest.raises(FileNotFoundError, match="cam0.png"):
        rapv2.generate_rapv2_csvs(tmp_path)
