#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ``list_attr_lfwa.txt`` 转成 train.csv 和 test.csv。不下载属性文件。

属性文件放在 ``datasets/LFWA/lfw/list_attr_lfwa.txt``。CSV 写到同一目录。
两个 CSV 都已存在则跳过，但仍检查属性文件。

示例::

    python datasets/LFWA/lfwa_attr_to_csv.py
    python datasets/LFWA/lfwa_attr_to_csv.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.lfwa import generate_lfwa_csvs

_DATASET_DIR = Path(__file__).resolve().parent / "lfw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFWA 属性表 → CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置 list_attr_lfwa.txt 和 CSV 的目录",
    )
    parser.add_argument(
        "--attr",
        type=Path,
        default=None,
        help="属性文件路径。默认是 dataset-dir/list_attr_lfwa.txt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="train.csv 和 test.csv 已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_lfwa_csvs(args.dataset_dir, attr_path=args.attr, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
