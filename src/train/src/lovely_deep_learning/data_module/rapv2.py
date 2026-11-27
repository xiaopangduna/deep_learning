"""RAP v2 行人裁剪图的属性表转 CSV。

图像在 ``RAP_dataset/``，标注在 ``rapv2_expand.pkl``。54 个实验属性和 44 个
颜色、遮挡属性写在同一张 CSV 里，实验属性在前。pkl 有 5 组训练 / 验证 / 测试
划分，默认写第 0 组。``Femal`` 里的 ``2`` 表示不确定，写成 ``0``。图已经是行人，
框是整张图。DataModule 用现成的 ``AttributeFromDetectBoxDataModule``，本模块不下载。
"""

from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from lovely_deep_learning.data_module.lfwa import attribute_column

_RELEASE_IMAGE_COUNT = 84_928
_RELEASE_FEMAL_UNKNOWN = 161
RAPV2_IMAGE_COUNT = _RELEASE_IMAGE_COUNT
RAPV2_ATTR_COUNT = 54
RAPV2_EXPAND_COUNT = 44
RAPV2_SPLIT_COUNT = 5
RAPV2_SPLIT_COUNTS = {"train": 50_957, "val": 16_986, "test": 16_985}
RAPV2_ANNOTATION_NAME = "rapv2_expand.pkl"
RAPV2_IMAGE_DIR = "RAP_dataset"
RAPV2_GENDER_ATTR = "Femal"
RAPV2_UNKNOWN_LABEL = 2
RAPV2_FEMAL_UNKNOWN = _RELEASE_FEMAL_UNKNOWN
RAPV2_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CSV_STEMS = ("train", "val", "test")


def generate_rapv2_csvs(
    dataset_dir,
    *,
    overwrite: bool = False,
    split_index: int = 0,
) -> Path:
    """写出 ``train.csv``、``val.csv``、``test.csv``。

    每张表先放 54 个实验属性，再放 44 个颜色和遮挡属性。``path_img`` 相对返回的
    数据目录。三个 CSV 都在且 ``overwrite`` 为 False 时跳过写入，但仍检查 pkl
    和第 ``split_index`` 组划分。换划分时要加 ``overwrite``，文件名里不带组号。
    """
    root = Path(dataset_dir).expanduser().resolve()
    annotation = load_rapv2_annotation(root / RAPV2_ANNOTATION_NAME)
    grouped = split_indexes(annotation["partition"], annotation["image_count"], split_index)
    outputs = _csv_paths(root)
    if not overwrite and all(path.is_file() for path in outputs.values()):
        print("RAP v2 CSV 已在，跳过生成:")
        for path in outputs.values():
            print(f"  - {path}")
        return root

    columns = annotation["attr_columns"] + annotation["expand_columns"]
    if len(columns) != len(set(columns)):
        raise RuntimeError("实验属性和颜色、遮挡属性的列名重复")
    label = np.concatenate([annotation["label"], annotation["expand_label"]], axis=0)
    sizes = _image_sizes(root, annotation["filenames"])
    _write_split_csvs(root, annotation["filenames"], sizes, columns, label, grouped)
    return root


def load_rapv2_annotation(path: Path) -> dict:
    """读取 ``rapv2_expand.pkl``。

    返回文件名、两组属性列名，以及形状 ``(属性, 图像)`` 的 0/1 标签。
    ``2`` 只允许出现在 ``Femal``，并改成 ``0``。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 {path}。v2 标注放 OpenPAR 的 rapv2_expand.pkl。"
            "目录里的 RAP_annotation.zip 是 RAP v1。"
        )
    payload = _load_pickle(path)
    filenames = _filenames(payload, path)
    attr_names = _name_list(payload, "attr_name", RAPV2_ATTR_COUNT, path)
    expand_names = _phrase_list(payload, RAPV2_EXPAND_COUNT, path)
    label = _binary_label(
        payload,
        "label",
        filenames,
        attr_names,
        path,
        allow_unknown_on=RAPV2_GENDER_ATTR,
    )
    expand_label = _binary_label(
        payload,
        "expand_label",
        filenames,
        expand_names,
        path,
        allow_unknown_on=None,
    )
    partition = payload.get("partition")
    if not isinstance(partition, dict):
        raise RuntimeError(f"{path} 缺少 partition")
    return {
        "filenames": filenames,
        "attr_columns": _columns(attr_names, path),
        "expand_columns": _columns(expand_names, path),
        "label": label,
        "expand_label": expand_label,
        "partition": partition,
        "image_count": len(filenames),
    }


def split_indexes(
    partition: dict,
    image_count: int,
    split_index: int,
) -> dict[str, np.ndarray]:
    """取出一组划分的图像下标。三组互斥，并且盖住全部图像。"""
    if split_index < 0 or split_index >= RAPV2_SPLIT_COUNT:
        raise RuntimeError(
            f"split_index 应在 0 到 {RAPV2_SPLIT_COUNT - 1} 之间，收到 {split_index}"
        )
    indexes: dict[str, np.ndarray] = {}
    pieces: list[np.ndarray] = []
    for stem, expected in RAPV2_SPLIT_COUNTS.items():
        raw = partition.get(stem)
        if not isinstance(raw, (list, tuple)) or len(raw) != RAPV2_SPLIT_COUNT:
            raise RuntimeError(f"partition[{stem!r}] 应有 {RAPV2_SPLIT_COUNT} 组下标")
        index = np.asarray(raw[split_index], dtype=np.int64).reshape(-1)
        if index.size != expected:
            raise RuntimeError(f"{stem} 应有 {expected} 张，收到 {index.size} 张")
        if index.size and (int(index.min()) < 0 or int(index.max()) >= image_count):
            raise RuntimeError(f"{stem} 的图像下标超出 0 到 {image_count - 1}")
        if len(set(index.tolist())) != index.size:
            raise RuntimeError(f"{stem} 的图像下标重复")
        indexes[stem] = index
        pieces.append(index)
    covered = np.concatenate(pieces) if pieces else np.array([], dtype=np.int64)
    if covered.size != image_count or len(set(covered.tolist())) != image_count:
        raise RuntimeError("训练、验证、测试没有正好盖住全部图像")
    return indexes


def png_size(path: Path) -> tuple[int, int]:
    """读 PNG 的 IHDR 宽高，不解码像素。返回 ``(宽, 高)``。"""
    with path.open("rb") as handle:
        if handle.read(8) != RAPV2_PNG_SIGNATURE:
            raise RuntimeError(f"不是 PNG: {path}")
        length_bytes = handle.read(4)
        tag = handle.read(4)
        payload = handle.read(8)
    if tag != b"IHDR" or len(length_bytes) < 4 or len(payload) < 8:
        raise RuntimeError(f"PNG 读不到宽高: {path}")
    width = int.from_bytes(payload[:4], "big")
    height = int.from_bytes(payload[4:8], "big")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"PNG 宽高非法: {path}")
    return width, height


def image_relpath(filename: str) -> str:
    """``CAM01-....png`` → ``RAP_dataset/CAM01-....png``。"""
    name = Path(filename).name
    if name != filename or Path(name).suffix.lower() != ".png":
        raise RuntimeError(f"图像名应为不带目录的 png，收到 {filename!r}")
    return f"{RAPV2_IMAGE_DIR}/{name}"


def _load_pickle(path: Path):
    _install_easydict()
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f"无法读取 {path}: {exc}") from exc


class _EasyDict(dict):
    """没有安装 easydict 时，用来接住 pkl 里的 EasyDict。"""


def _install_easydict() -> None:
    """pkl 由 easydict.EasyDict 序列化。没有装这个包时用普通 dict 接住。"""
    module = sys.modules.get("easydict")
    if module is not None and hasattr(module, "EasyDict"):
        return
    module = types.ModuleType("easydict")
    _EasyDict.__module__ = "easydict"
    _EasyDict.__qualname__ = "EasyDict"
    module.EasyDict = _EasyDict
    sys.modules["easydict"] = module


def _filenames(payload, path: Path) -> list[str]:
    stored = payload.get("image_name")
    if stored is None:
        raise RuntimeError(f"{path} 缺少 image_name")
    filenames = [str(name) for name in stored]
    if len(filenames) != RAPV2_IMAGE_COUNT:
        raise RuntimeError(f"{path} 应有 {RAPV2_IMAGE_COUNT} 张图，收到 {len(filenames)} 张")
    if len(filenames) != len(set(filenames)):
        raise RuntimeError(f"{path} 图像名重复")
    for name in filenames:
        image_relpath(name)
    return filenames


def _name_list(payload, key: str, expected: int, path: Path) -> list[str]:
    raw = payload.get(key)
    if raw is None:
        raise RuntimeError(f"{path} 缺少 {key}")
    names = [str(name) for name in raw]
    if len(names) != expected:
        raise RuntimeError(f"{path} 的 {key} 应有 {expected} 个，收到 {len(names)} 个")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path} 的 {key} 有重复名字")
    return names


def _phrase_list(payload, expected: int, path: Path) -> list[str]:
    """44 个扩展属性用英文短语做列名。

    pkl 里的 ``expand_attr_name`` 从鞋子颜色和遮挡开始与短语对不上，短语本身是齐的。
    """
    raw = payload.get("expand_attributes")
    if raw is None:
        raise RuntimeError(f"{path} 缺少 expand_attributes")
    names = [str(name) for name in raw]
    if len(names) != expected:
        raise RuntimeError(f"{path} 的扩展属性应有 {expected} 个，收到 {len(names)} 个")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path} 的扩展属性名重复")
    return names


def _binary_label(
    payload,
    key: str,
    filenames: list[str],
    names: list[str],
    path: Path,
    *,
    allow_unknown_on: str | None,
) -> np.ndarray:
    raw = payload.get(key)
    if raw is None:
        raise RuntimeError(f"{path} 缺少 {key}")
    label = np.asarray(raw)
    expected = (len(filenames), len(names))
    if label.shape != expected:
        raise RuntimeError(f"{path} 的 {key} 形状是 {label.shape}，期望 {expected}")
    if not np.isin(label, (0, 1, RAPV2_UNKNOWN_LABEL)).all():
        raise RuntimeError(f"{path} 的 {key} 只能是 0、1 或 2")
    unknown = label == RAPV2_UNKNOWN_LABEL
    allowed = np.zeros(len(names), dtype=bool)
    if allow_unknown_on is not None:
        if allow_unknown_on not in names:
            raise RuntimeError(f"{path} 缺少属性 {allow_unknown_on}")
        allowed[names.index(allow_unknown_on)] = True
        if len(filenames) == _RELEASE_IMAGE_COUNT:
            found = int(unknown[:, names.index(allow_unknown_on)].sum())
            if found != _RELEASE_FEMAL_UNKNOWN:
                raise RuntimeError(
                    f"{allow_unknown_on} 的不确定标注应有 {_RELEASE_FEMAL_UNKNOWN} 个，收到 {found} 个"
                )
    bad = [names[index] for index in range(len(names)) if unknown[:, index].any() and not allowed[index]]
    if bad:
        raise RuntimeError(f"{path} 的 {key} 里这些属性出现了 2: {bad}")
    binary = np.array(label, dtype=np.int64, copy=True)
    binary[unknown] = 0
    return binary.T


def _columns(names: list[str], path: Path) -> list[str]:
    columns = [attribute_column(name) for name in names]
    if len(columns) != len(set(columns)):
        raise RuntimeError(f"{path} 属性列名重复")
    return columns


def _csv_paths(root: Path) -> dict[str, Path]:
    return {stem: root / f"{stem}.csv" for stem in _CSV_STEMS}


def _image_sizes(root: Path, filenames: list[str]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    step = max(len(filenames) // 10, 1)
    for index, filename in enumerate(filenames):
        path = root / image_relpath(filename)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到行人图: {path}。先把 RAP_dataset.zip 解到 {root / RAPV2_IMAGE_DIR}。"
            )
        sizes.append(png_size(path))
        done = index + 1
        if done == len(filenames) or done % step == 0:
            print(f"  读取 PNG 宽高 {done}/{len(filenames)}")
    return sizes


def _write_split_csvs(
    root: Path,
    filenames: list[str],
    sizes: list[tuple[int, int]],
    columns: list[str],
    label: np.ndarray,
    grouped: dict[str, np.ndarray],
) -> None:
    for stem, index in grouped.items():
        dest = root / f"{stem}.csv"
        _write_attribute_csv(dest, filenames, sizes, columns, label, index)
        print(f"写出 {dest}，{len(index)} 行")


def _write_attribute_csv(
    dest: Path,
    filenames: list[str],
    sizes: list[tuple[int, int]],
    columns: list[str],
    label: np.ndarray,
    index: np.ndarray,
) -> None:
    chosen = index.tolist()
    frame = pd.DataFrame(
        {
            "path_img": [image_relpath(filenames[i]) for i in chosen],
            "cx": "0.5",
            "cy": "0.5",
            "w": "1",
            "h": "1",
            "img_w": [sizes[i][0] for i in chosen],
            "img_h": [sizes[i][1] for i in chosen],
        }
    )
    for column_index, column in enumerate(columns):
        frame[column] = label[column_index, index]
    frame.to_csv(dest, index=False)
