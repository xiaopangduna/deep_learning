#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CelebA 属性表、划分和身份写出训练 CSV。

数据目录里要有 ``Anno/``、``Eval/`` 和 ``Img/img_align_celeba/``。
浏览器整包多一层 ``CelebA/`` 时会自己走进去。三个 CSV 都已存在则跳过写入，
但仍检查这三份文本。

示例::

    python datasets/CelebA/celeba_attr_to_csv.py
    python datasets/CelebA/celeba_attr_to_csv.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.celeba import generate_celeba_csvs

_DATASET_DIR = Path(__file__).resolve().parent / "celeba"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA 属性表 → CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置 Anno、Eval、Img 的目录；多一层 CelebA/ 时也会认",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="三个 CSV 已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_celeba_csvs(args.dataset_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
