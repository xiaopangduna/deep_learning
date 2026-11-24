#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载并解压原始 LFW 图像包。不生成 CSV，也不读取属性标签。

本脚本放在 ``datasets/LFWA/``。压缩包和解压结果在同级的 ``lfw/``（可以是软链接）。
已经有完整的 ``lfw.tgz`` 就跳过下载；已经解开 13233 张 JPEG 就跳过解压。

示例::

    python datasets/LFWA/download_lfw.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.lfwa import LFW_TGZ_URL, prepare_lfw_images

_DATASET_DIR = Path(__file__).resolve().parent / "lfw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并解压原始 LFW 图像")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="放置 lfw.tgz 和 lfw/ 的目录",
    )
    parser.add_argument(
        "--url",
        default=LFW_TGZ_URL,
        help="lfw.tgz 直链。默认是网盘文件的下载地址，不是分享页",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_lfw_images(args.dataset_dir, url=args.url)


if __name__ == "__main__":
    main()
