#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解开已经放好的 LFWA+ 整包。不下载。

整包是网盘目录一次下载得到的 ``LFWA+-*.zip``，放在 ``datasets/LFWA/lfw/``。
文件名里的时间每次下载都会变。三个 ``.mat`` 解到同一目录；图像已经有 13233 张就不再解开里面的 ``lfw.zip``。

示例::

    python datasets/LFWA/unpack_lfwa_plus.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.lfwa import unpack_lfwa_plus

_DATASET_DIR = Path(__file__).resolve().parent / "lfw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解开本地的 LFWA+ 整包")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置 LFWA+-*.zip、属性文件和图像的目录",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="整包路径。默认使用 dataset-dir 里唯一的 LFWA+-*.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unpack_lfwa_plus(args.dataset_dir, archive=args.archive)


if __name__ == "__main__":
    main()
