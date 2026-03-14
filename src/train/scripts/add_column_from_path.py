#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据路径列生成新列，并写回 CSV。

示例：

输入 CSV:
path_img
imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG

运行：

python add_column_from_path.py data.csv \
    --path-col path_img \
    --pos -2 \
    --col-name class

输出 CSV:

path_img,class
imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG,n01440764
"""

import argparse
import csv
import sys
from pathlib import Path


def extract_from_path(path_str: str, pos: int) -> str:
    """根据路径位置提取字符串"""
    parts = Path(path_str).parts
    return parts[pos]


def main():

    parser = argparse.ArgumentParser(
        description="根据路径列提取字段并新增一列"
    )

    parser.add_argument(
        "csv_file",
        help="CSV 文件路径"
    )

    parser.add_argument(
        "--path-col",
        required=True,
        help="路径列名，例如 path_img"
    )

    parser.add_argument(
        "--pos",
        type=int,
        required=True,
        help="路径位置，例如 -2"
    )

    parser.add_argument(
        "--col-name",
        required=True,
        help="新列名，例如 class"
    )

    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV 分隔符"
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)

    if not csv_path.is_file():
        print(f"错误：文件不存在 {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(csv_path, "r", encoding=args.encoding) as f:
            reader = csv.DictReader(f, delimiter=args.delimiter)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except Exception as e:
        print(f"读取 CSV 失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.path_col not in fieldnames:
        print(f"错误：列 {args.path_col} 不存在", file=sys.stderr)
        sys.exit(1)

    if args.col_name not in fieldnames:
        fieldnames.append(args.col_name)

    new_rows = []

    for row in rows:

        path_str = row[args.path_col]

        try:
            value = extract_from_path(path_str, args.pos)
        except Exception as e:
            print(f"警告：处理路径失败 {path_str}: {e}", file=sys.stderr)
            value = ""

        row[args.col_name] = value

        new_rows.append(row)

    try:
        with open(csv_path, "w", encoding=args.encoding, newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=args.delimiter
            )

            writer.writeheader()
            writer.writerows(new_rows)

    except Exception as e:
        print(f"写入 CSV 失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"完成：已更新 {csv_path}")


if __name__ == "__main__":
    main()