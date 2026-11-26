"""CelebA 对齐脸的属性文本转 CSV。

标注在 ``Anno/``、``Eval/``，对齐 JPEG 在 ``Img/img_align_celeba/``。
官方 ``list_eval_partition.txt`` 的 ``0`` / ``1`` / ``2`` 写成
``train.csv`` / ``val.csv`` / ``test.csv``。属性取值 ``1`` / ``-1`` 写成 ``1`` / ``0``。
这些图已经是脸，框是整张图。DataModule 用现成的
``AttributeFromDetectBoxDataModule``，本模块不下载。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lovely_deep_learning.data_module.lfwa import attribute_column, jpeg_size

CELEBA_IMAGE_COUNT = 202_599
CELEBA_ATTR_COUNT = 40
CELEBA_IDENTITY_COUNT = 10_177
CELEBA_SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
CELEBA_SPLIT_COUNTS = {"train": 162_770, "val": 19_867, "test": 19_962}
CELEBA_ATTR_FILE = Path("Anno/list_attr_celeba.txt")
CELEBA_PARTITION_FILE = Path("Eval/list_eval_partition.txt")
CELEBA_IDENTITY_FILE = Path("Anno/identity_CelebA.txt")
CELEBA_IMAGE_DIR = Path("Img/img_align_celeba")
CELEBA_IMAGE_SIZE = (178, 218)


def resolve_celeba_root(dataset_dir) -> Path:
    """找到同时放着 ``Anno/``、``Eval/``、``Img/`` 的目录。

    ``dataset_dir`` 自己有属性表就用它。浏览器整包解开后多一层 ``CelebA/`` 时，用里面那层。
    """
    path = Path(dataset_dir).expanduser()
    if path.is_symlink() and not path.exists():
        raise FileNotFoundError(f"CelebA dataset_dir 不存在: {path}")
    root = path.resolve()
    if _has_annotations(root):
        return root
    nested = root / "CelebA"
    if _has_annotations(nested):
        return nested
    raise FileNotFoundError(
        f"在 {root} 和 {nested} 都找不到 {CELEBA_ATTR_FILE}。"
        "先按 datasets/CelebA/README.md 解开网盘整包和对齐 JPEG。"
    )


def generate_celeba_csvs(dataset_dir, *, overwrite: bool = False) -> Path:
    """写出 40 个属性的 ``train.csv``、``val.csv``、``test.csv``。

    ``path_img`` 相对返回的数据目录。三个 CSV 都在且 ``overwrite`` 为 False 时跳过写入，
    但仍检查属性表、划分和身份。返回实际写入的数据目录。
    """
    root = resolve_celeba_root(dataset_dir)
    names, filenames, label = load_attribute_table(root / CELEBA_ATTR_FILE)
    partition = load_partition(root / CELEBA_PARTITION_FILE)
    identity = load_identity(root / CELEBA_IDENTITY_FILE)
    grouped = align_celeba_tables(filenames, partition, identity)
    outputs = {stem: root / f"{stem}.csv" for stem in CELEBA_SPLIT_COUNTS}
    if not overwrite and all(path.is_file() for path in outputs.values()):
        print("CelebA CSV 已在，跳过生成:")
        for path in outputs.values():
            print(f"  - {path}")
        return root

    columns = [attribute_column(name) for name in names]
    if len(columns) != len(set(columns)):
        raise RuntimeError("40 个属性列名重复")
    sizes = _image_sizes(root, filenames)
    for code, stem in CELEBA_SPLIT_NAMES.items():
        index = grouped[code]
        dest = outputs[stem]
        _write_attribute_csv(dest, filenames, sizes, columns, label, index)
        print(f"写出 {dest}，{len(index)} 行")
    return root


def load_attribute_table(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """读取属性表。返回属性名、文件名，以及形状 ``(属性, 图像)`` 的 0/1 标签。"""
    lines = _nonempty_lines(path)
    if len(lines) < 3:
        raise RuntimeError(f"{path} 至少要有图像数、属性名和一行标注")
    declared = _declared_count(path, lines[0])
    names = lines[1].split()
    if len(names) != CELEBA_ATTR_COUNT:
        raise RuntimeError(f"{path} 应有 {CELEBA_ATTR_COUNT} 个属性，收到 {len(names)} 个")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path} 属性名重复")
    filenames: list[str] = []
    label = np.empty((CELEBA_ATTR_COUNT, len(lines) - 2), dtype=np.int64)
    for row_index, line in enumerate(lines[2:]):
        parts = line.split()
        if len(parts) != 1 + CELEBA_ATTR_COUNT:
            raise RuntimeError(
                f"{path}:{row_index + 3} 应有文件名和 {CELEBA_ATTR_COUNT} 个值"
            )
        filenames.append(_image_filename(parts[0], path))
        for column, raw in enumerate(parts[1:]):
            if raw == "1":
                label[column, row_index] = 1
            elif raw == "-1":
                label[column, row_index] = 0
            else:
                raise RuntimeError(f"{path}:{row_index + 3} 属性值只能是 1 或 -1，收到 {raw}")
    if declared != len(filenames):
        raise RuntimeError(f"{path} 声明 {declared} 行，实际 {len(filenames)} 行")
    if len(filenames) != len(set(filenames)):
        raise RuntimeError(f"{path} 图像名重复")
    return names, filenames, label


def load_partition(path: Path) -> dict[str, int]:
    """读取划分。文件名映射到 ``0`` 训练、``1`` 验证、``2`` 测试。"""
    partition: dict[str, int] = {}
    for line_no, line in enumerate(_nonempty_lines(path), start=1):
        parts = line.split()
        if len(parts) != 2 or parts[1] not in {"0", "1", "2"}:
            raise RuntimeError(f"{path}:{line_no} 应为 <图像名> 0、1 或 2")
        filename = _image_filename(parts[0], path)
        if filename in partition:
            raise RuntimeError(f"{path} 图像名重复: {filename}")
        partition[filename] = int(parts[1])
    return partition


def load_identity(path: Path) -> dict[str, str]:
    """读取身份。文件名映射到身份编号。"""
    identity: dict[str, str] = {}
    for line_no, line in enumerate(_nonempty_lines(path), start=1):
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            raise RuntimeError(f"{path}:{line_no} 应为 <图像名> <身份编号>")
        filename = _image_filename(parts[0], path)
        if filename in identity:
            raise RuntimeError(f"{path} 图像名重复: {filename}")
        identity[filename] = parts[1]
    return identity


def align_celeba_tables(
    filenames: list[str],
    partition: dict[str, int],
    identity: dict[str, str],
) -> dict[int, np.ndarray]:
    """对齐三张表。返回各划分在属性表里的下标，并检查人数和身份不跨划分。"""
    names = set(filenames)
    if names != set(partition) or names != set(identity):
        raise RuntimeError("属性表、划分表和身份表的图像名不一致")
    if len(identity) != CELEBA_IMAGE_COUNT:
        raise RuntimeError(f"应有 {CELEBA_IMAGE_COUNT} 张图，收到 {len(identity)} 张")
    if len(set(identity.values())) != CELEBA_IDENTITY_COUNT:
        raise RuntimeError(
            f"应有 {CELEBA_IDENTITY_COUNT} 个身份，收到 {len(set(identity.values()))} 个"
        )
    grouped: dict[int, list[int]] = {0: [], 1: [], 2: []}
    people: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for index, filename in enumerate(filenames):
        code = partition[filename]
        grouped[code].append(index)
        people[code].add(identity[filename])
    for left, right in ((0, 1), (0, 2), (1, 2)):
        shared = people[left] & people[right]
        if shared:
            raise RuntimeError(
                f"划分 {CELEBA_SPLIT_NAMES[left]} 和 {CELEBA_SPLIT_NAMES[right]} "
                f"有 {len(shared)} 个相同身份"
            )
    indexes: dict[int, np.ndarray] = {}
    for code, stem in CELEBA_SPLIT_NAMES.items():
        expected = CELEBA_SPLIT_COUNTS[stem]
        if len(grouped[code]) != expected:
            raise RuntimeError(f"{stem} 应有 {expected} 张，收到 {len(grouped[code])} 张")
        indexes[code] = np.asarray(grouped[code], dtype=np.int64)
    return indexes


def image_relpath(filename: str) -> str:
    """``000001.jpg`` → ``Img/img_align_celeba/000001.jpg``。"""
    return f"{CELEBA_IMAGE_DIR.as_posix()}/{_image_filename(filename, Path('filename'))}"


def _image_sizes(root: Path, filenames: list[str]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    step = max(len(filenames) // 10, 1)
    for index, filename in enumerate(filenames):
        path = root / CELEBA_IMAGE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到对齐 JPEG: {path}。先解开 Img/img_align_celeba.zip。"
            )
        size = jpeg_size(path)
        if size != CELEBA_IMAGE_SIZE:
            raise RuntimeError(
                f"{path} 宽高是 {size[0]}×{size[1]}，期望 "
                f"{CELEBA_IMAGE_SIZE[0]}×{CELEBA_IMAGE_SIZE[1]}"
            )
        sizes.append(size)
        done = index + 1
        if done == len(filenames) or done % step == 0:
            print(f"  读取 JPEG 宽高 {done}/{len(filenames)}")
    return sizes


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


def _has_annotations(root: Path) -> bool:
    return (root / CELEBA_ATTR_FILE).is_file()


def _nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _declared_count(path: Path, text: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise RuntimeError(f"{path} 第一行应是图像数，收到 {text!r}") from exc


def _image_filename(filename: str, path: Path) -> str:
    name = Path(filename).name
    stem = name[:-4]
    if not name.lower().endswith(".jpg") or len(stem) != 6 or not stem.isdigit():
        raise RuntimeError(f"{path} 图像名应为 000001.jpg 这种 6 位编号: {filename}")
    return name
