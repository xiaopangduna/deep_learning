#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从三个 ``.mat`` 写出 40 属性和 73 属性的 CSV。

``lfw_att_40.mat``、``lfw_att_73.mat`` 和 ``indices_train_test.mat`` 放在
``datasets/LFWA/lfw/``。图像目录是同一层的 ``lfw/``。两套属性各写
``train``、``val``、``test`` 三张表。六个 CSV 都已存在则跳过写入，但仍检查
``.mat``。

示例::

    uv run --with h5py python datasets/LFWA/lfwa_attr_to_csv.py
    uv run --with h5py python datasets/LFWA/lfwa_attr_to_csv.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.lfwa import generate_lfwa_csvs

_DATASET_DIR = Path(__file__).resolve().parent / "lfw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFWA 的 .mat 属性表 → CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置三个 .mat、图像目录和 CSV 的目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="六个 CSV 已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_lfwa_csvs(args.dataset_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
