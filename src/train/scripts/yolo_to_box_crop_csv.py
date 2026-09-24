#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 YOLO 检测 CSV 按框展开成分类 CSV。原图不复制，不筛选类别，不改 class_id。

源 CSV 列：``path_img``, ``path_label_detect_yolo``。路径相对源 CSV 所在目录。
写出目录默认 ``datasets/COCO_class``：

- ``train.csv`` / ``val.csv`` / ``test.csv``：
  ``path_img,class_name,class_id,cx,cy,w,h,img_w,img_h``
- ``predict.csv``：``path_img,cx,cy,w,h,img_w,img_h``（与 val 同一批框，无类别）
- ``map_class_id_to_class_name.csv``：COCO 80 类原始 id

``class_id`` 是 YOLO / COCO 原始 id。``img_w,img_h`` 供 Dataset 做短边过滤，避免每次训练重开图像。
类别、无效框、短边过滤在 ``BoxCropClassifierDataset`` 初始化时完成。``test.csv`` 与 ``val.csv`` 相同。

示例::

python scripts/yolo_to_box_crop_csv.py \
    --train-csv datasets/COCO/train.csv \
    --val-csv datasets/COCO/val.csv \
    --out-dir datasets/COCO_class
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from lovely_deep_learning.data_module.coco import COCO80_CLASS_NAMES

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
PREDICT_COLUMNS = ["path_img", "cx", "cy", "w", "h", "img_w", "img_h"]


def _resolve(csv_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (csv_path.parent / path).resolve()


def _image_size(path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    """返回 ``(width, height)``。只读文件头，同一张图只打开一次。"""
    if path not in cache:
        with Image.open(path) as im:
            cache[path] = im.size
    return cache[path]


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO80_CLASS_NAMES):
        return COCO80_CLASS_NAMES[class_id]
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


def boxes_from_detection_csv(detection_csv: Path) -> list[dict[str, object]]:
    """展开一张检测 CSV。``path_img`` 仍是绝对路径，由调用方改成相对输出目录。"""
    detection_csv = detection_csv.expanduser().resolve()
    table = pd.read_csv(detection_csv, dtype=str, keep_default_na=False)
    size_cache: dict[Path, tuple[int, int]] = {}
    out: list[dict[str, object]] = []
    for record in table.to_dict(orient="records"):
        img_path = _resolve(detection_csv, record["path_img"])
        label_path = _resolve(detection_csv, record["path_label_detect_yolo"])
        if not img_path.is_file():
            raise FileNotFoundError(f"图像不存在：{img_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"标签不存在：{label_path}")
        width, height = _image_size(img_path, size_cache)
        for class_id, cx, cy, w, h in _read_yolo_rows(label_path):
            out.append(
                {
                    "path_img_abs": img_path,
                    "class_name": _class_name(class_id),
                    "class_id": class_id,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                    "img_w": width,
                    "img_h": height,
                }
            )
    return out


def _relativize(rows: list[dict[str, object]], out_dir: Path) -> list[dict[str, object]]:
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
    return relativized


def write_box_crop_csvs(
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rel = _relativize(train_rows, out_dir)
    val_rel = _relativize(val_rows, out_dir)
    pd.DataFrame(train_rel, columns=LABEL_COLUMNS).to_csv(
        out_dir / "train.csv", index=False
    )
    pd.DataFrame(val_rel, columns=LABEL_COLUMNS).to_csv(out_dir / "val.csv", index=False)
    pd.DataFrame(val_rel, columns=LABEL_COLUMNS).to_csv(out_dir / "test.csv", index=False)
    pd.DataFrame(val_rel, columns=PREDICT_COLUMNS).to_csv(
        out_dir / "predict.csv", index=False
    )
    pd.DataFrame(
        {
            "class_id": list(range(len(COCO80_CLASS_NAMES))),
            "class_name": list(COCO80_CLASS_NAMES),
        }
    ).to_csv(out_dir / "map_class_id_to_class_name.csv", index=False)


def convert(train_csv: Path, val_csv: Path, out_dir: Path) -> None:
    train_rows = boxes_from_detection_csv(train_csv)
    val_rows = boxes_from_detection_csv(val_csv)
    out_dir = out_dir.expanduser().resolve()
    write_box_crop_csvs(train_rows, val_rows, out_dir)
    print(
        f"写出 {out_dir} ：train {len(train_rows)} 框，val/test/predict {len(val_rows)} 框"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 检测 CSV → 框级分类 CSV（保留全部类别，不复制图像）"
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/COCO_class"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.train_csv, args.val_csv, args.out_dir)


if __name__ == "__main__":
    main()
