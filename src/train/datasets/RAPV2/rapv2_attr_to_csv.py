#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ``rapv2_expand.pkl`` 写出 RAP v2 的训练 CSV。

图像目录是 ``datasets/RAPV2/RAP_dataset/``。54 个实验属性和 44 个颜色、遮挡属性
写在同一张 ``train.csv``、``val.csv``、``test.csv`` 里，实验属性在前。三个 CSV
都已存在则跳过写入，但仍检查 pkl。

示例::

    python datasets/RAPV2/rapv2_attr_to_csv.py
    python datasets/RAPV2/rapv2_attr_to_csv.py --split-index 1 --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.rapv2 import generate_rapv2_csvs

_DATASET_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAP v2 属性 pkl → CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置 rapv2_expand.pkl、RAP_dataset 和 CSV 的目录",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="使用 pkl 里的第几组划分，0 到 4。默认 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="三个 CSV 已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_rapv2_csvs(
        args.dataset_dir,
        overwrite=args.overwrite,
        split_index=args.split_index,
    )


if __name__ == "__main__":
    main()
