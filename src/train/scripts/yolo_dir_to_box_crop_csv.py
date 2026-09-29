#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YOLO 目录按框展开成一张分类 CSV。原图不复制，不筛选类别，不改 class_id。

目录布局::

    <data-root>/dataset.yaml          # names
    <data-root>/images/<split>/*
    <data-root>/labels/<split>/*.txt

``--split`` 省略且 ``images`` 下只有一个子目录时，用该子目录。
空标签（负样本图）不写行。``path_img`` 相对输出 CSV 所在目录；符号链接先解析到真实文件。
类别、无效框、短边过滤在 ``BoxCropClassifierDataset`` 初始化时完成。

示例::

python scripts/yolo_dir_to_box_crop_csv.py \\
    --data-root /home/huangwenhua/project/dataset/head/v011_head_age \\
    --split train_v011 \\
    --out datasets/head_age/train_v011.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

LABEL_COLUMNS = [
    "path_img",
    "class_name",
    "class_id",
    "cx",
    "cy",
    "w",
    "h",
    "img_w",
    "img_h",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _image_size(path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    """返回 ``(width, height)``。只读文件头，同一张图只打开一次。"""
    if path not in cache:
        with Image.open(path) as im:
            cache[path] = im.size
    return cache[path]


def _class_name(class_id: int, names: dict[int, str]) -> str:
    if class_id in names:
        return names[class_id]
    return str(class_id)


def _read_yolo_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        rows.append((class_id, cx, cy, w, h))
    return rows


def load_class_names(data_root: Path) -> dict[int, str]:
    """读取 ``dataset.yaml`` 的 ``names``。支持 id 字典和列表两种写法。"""
    yaml_path = data_root / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"缺少类别文件：{yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, list):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    raise ValueError(f"{yaml_path} 的 names 须为字典或列表，收到 {type(names).__name__}")


def resolve_split(data_root: Path, split: str | None) -> str:
    """定位 ``images/<split>`` 与 ``labels/<split>``。"""
    images_root = data_root / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"缺少图像目录：{images_root}")
    if split is None:
        subs = sorted(path.name for path in images_root.iterdir() if path.is_dir())
        if len(subs) != 1:
            raise ValueError(f"请用 --split 指定划分，images 下有 {subs}")
        split = subs[0]
    images_dir = images_root / split
    labels_dir = data_root / "labels" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"缺少图像划分：{images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"缺少标签划分：{labels_dir}")
    return split


def _index_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() and not path.is_symlink():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if path.stem in found:
            raise ValueError(f"重复 stem：{path.name} 与 {found[path.stem].name}")
        found[path.stem] = path
    return found


def _paired_files(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    images = _index_by_stem(images_dir, IMAGE_SUFFIXES)
    labels = _index_by_stem(labels_dir, {".txt"})
    image_stems = set(images)
    label_stems = set(labels)
    if image_stems != label_stems:
        only_images = sorted(image_stems - label_stems)
        only_labels = sorted(label_stems - image_stems)
        raise FileNotFoundError(
            "图像与标签 stem 不一致："
            f"仅图像 {len(only_images)} 个 {only_images[:5]}，"
            f"仅标签 {len(only_labels)} 个 {only_labels[:5]}"
        )
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def boxes_from_yolo_split(
    data_root: Path,
    split: str,
    names: dict[int, str],
) -> tuple[list[dict[str, object]], int]:
    """展开一个划分。返回框行和空标签张数。``path_img_abs`` 已解析符号链接。"""
    pairs = _paired_files(data_root / "images" / split, data_root / "labels" / split)
    size_cache: dict[Path, tuple[int, int]] = {}
    rows: list[dict[str, object]] = []
    n_empty = 0
    for image_path, label_path in pairs:
        resolved = image_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"图像不存在：{image_path} -> {resolved}")
        if not label_path.is_file():
            raise FileNotFoundError(f"标签不存在：{label_path}")
        yolo_rows = _read_yolo_rows(label_path)
        if not yolo_rows:
            n_empty += 1
            continue
        width, height = _image_size(resolved, size_cache)
        for class_id, cx, cy, w, h in yolo_rows:
            rows.append(
                {
                    "path_img_abs": resolved,
                    "class_name": _class_name(class_id, names),
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "img_w": width,
                    "img_h": height,
                }
            )
    return rows, n_empty


def write_box_crop_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    relativized = []
    for row in rows:
        rel = Path(os.path.relpath(row["path_img_abs"], out_dir)).as_posix()
        relativized.append(
            {
                "path_img": rel,
                "class_name": row["class_name"],
                "class_id": row["class_id"],
                "cx": row["cx"],
                "cy": row["cy"],
                "w": row["w"],
                "h": row["h"],
                "img_w": row["img_w"],
                "img_h": row["img_h"],
            }
        )
    pd.DataFrame(relativized, columns=LABEL_COLUMNS).to_csv(out_csv, index=False)


def convert(data_root: Path, out_csv: Path, split: str | None = None) -> None:
    data_root = data_root.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()
    names = load_class_names(data_root)
    split_name = resolve_split(data_root, split)
    rows, n_empty = boxes_from_yolo_split(data_root, split_name, names)
    write_box_crop_csv(rows, out_csv)
    print(f"写出 {out_csv} ：{len(rows)} 框，跳过空标签 {n_empty} 张（split={split_name}）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 目录 → 一张框级分类 CSV（保留全部类别，不复制图像）"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        default=None,
        help="images/<split> 与 labels/<split>。省略且只有一个子目录时自动使用",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.data_root, args.out, args.split)


if __name__ == "__main__":
    main()
