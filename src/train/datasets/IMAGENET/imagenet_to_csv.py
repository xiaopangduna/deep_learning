#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 ILSVRC ImageSets 生成 ImageNet 的 train/val/test/predict CSV。不下载数据。

本脚本放在 ``datasets/IMAGENET/``。压缩包、解压结果和 CSV 在同级的 ``imagenet/``（可以是软链接）。
默认把 CSV 写到该目录。写入逻辑与 ``ImageNetDataModule.prepare_data`` 相同。四个 CSV 已存在则跳过。

示例::

python datasets/IMAGENET/imagenet_to_csv.py
python datasets/IMAGENET/imagenet_to_csv.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lovely_deep_learning.data_module.image_net import generate_imagenet_csvs

_DATASET_DIR = Path(__file__).resolve().parent / "imagenet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ILSVRC ImageSets → ImageNet CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DATASET_DIR,
        help="含压缩包和 ILSVRC/ 的目录，CSV 也写在这里",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="四个 CSV 已存在时仍重新生成",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_imagenet_csvs(args.dataset_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
