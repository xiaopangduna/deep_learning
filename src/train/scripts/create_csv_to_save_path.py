#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查找指定目录下指定扩展名的文件。

特点：
- 默认查找常见图像格式
- 扩展名匹配大小写不敏感
- 默认递归搜索
- 支持关闭递归
- 支持相对路径输出
- 支持 CSV 输出
- 支持自定义 CSV 表头
"""

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def normalize_extensions(exts):
    """统一扩展名格式（小写并确保带点）"""
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}


def find_files(folder: Path, extensions: set, recursive: bool):
    """查找匹配扩展名的文件"""
    iterator = folder.rglob("*") if recursive else folder.iterdir()

    for path in iterator:
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def main():
    example_text = """
示例目录结构:

dataset/
├── cat
│   ├── 001.jpg
│   └── 002.jpg
├── dog
│   ├── 001.jpg
│   └── 002.png
└── other.txt


示例 1：查找所有图像文件（默认）

    python find_files.py dataset

输出:

    /data/dataset/cat/001.jpg
    /data/dataset/cat/002.jpg
    /data/dataset/dog/001.jpg
    /data/dataset/dog/002.png


示例 2：指定扩展名

    python find_files.py dataset txt csv


示例 3：关闭递归

    python find_files.py dataset --no-recursive


示例 4：输出相对路径

    python find_files.py dataset --relative-to dataset

输出:

    cat/001.jpg
    cat/002.jpg
    dog/001.jpg
    dog/002.png


示例 5：输出 CSV 文件

    python find_files.py dataset -o files.csv --header path

生成 CSV:

    path
    /data/dataset/cat/001.jpg
    /data/dataset/cat/002.jpg
    /data/dataset/dog/001.jpg
    /data/dataset/dog/002.png
"""

    parser = argparse.ArgumentParser(
        description="查找指定扩展名的文件并输出路径列表",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("folder", help="搜索目录")

    parser.add_argument("extensions", nargs="*", help="文件扩展名（默认常见图像格式）")

    parser.add_argument("--no-recursive", action="store_true", help="仅搜索当前目录")

    parser.add_argument("--relative-to", metavar="DIR", help="输出相对于该目录的路径")

    parser.add_argument("-o", "--output", metavar="FILE", help="输出 CSV 文件")

    parser.add_argument("--header", help="CSV 表头")

    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.is_dir():
        print(f"错误：目录不存在 -> {folder}", file=sys.stderr)
        sys.exit(1)

    # 扩展名处理
    extensions = normalize_extensions(args.extensions) if args.extensions else DEFAULT_IMAGE_EXTENSIONS

    # relative 基准目录
    base_dir = None
    if args.relative_to:
        base_dir = Path(args.relative_to).resolve()
        if not base_dir.is_dir():
            print(f"错误：relative-to 目录不存在 -> {base_dir}", file=sys.stderr)
            sys.exit(1)

    recursive = not args.no_recursive

    try:
        files = list(find_files(folder, extensions, recursive))
    except Exception as e:
        print(f"搜索失败: {e}", file=sys.stderr)
        sys.exit(1)

    files.sort()

    # 生成输出路径
    out_paths = []

    for f in files:
        p = f.resolve()

        if base_dir:
            try:
                p = p.relative_to(base_dir)
            except ValueError:
                pass

        out_paths.append(str(p))

    # 输出 CSV
    if args.output:

        try:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if args.header:
                    writer.writerow([args.header])

                for p in out_paths:
                    writer.writerow([p])

        except Exception as e:
            print(f"写入 CSV 失败: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        for p in out_paths:
            print(p)


if __name__ == "__main__":
    main()
