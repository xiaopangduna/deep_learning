#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按图像列表生成 COCO 2017 的 train/val/test/predict CSV 和类别映射。不下载数据。

本脚本放在 ``datasets/COCO/``。压缩包、解压结果和 CSV 在同级的 ``coco/``（可以是软链接）。
默认把 CSV 写到该目录。写入逻辑与 ``generate_coco_csvs`` 相同，也就是 ``prepare_data`` 里写 CSV 的那一步。五个文件已存在则跳过。

示例::

python datasets/COCO/coco_to_csv.py
python datasets/COCO/coco_to_csv.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.coco import generate_coco_csvs

_DATASET_DIR = Path(__file__).resolve().parent / "coco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO YOLO labels → CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="含压缩包、images/ 和 labels/ 的目录，CSV 也写在这里",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="五个文件已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_coco_csvs(args.dataset_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
