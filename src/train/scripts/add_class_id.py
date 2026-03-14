#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为 CSV 文件添加 class_id 列，映射关系来自 YAML 文件。

该脚本会读取 CSV 文件中的类别列，根据 YAML 中定义的
{class_name: class_id} 映射，生成新的 class_id 列。

注意：
脚本会直接修改原 CSV 文件。

使用示例:
python add_class_id.py dataset.csv class_map.yaml
"""

import argparse
import csv
import yaml
import sys
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="根据 YAML 映射为 CSV 文件添加 class_id 列（直接修改原 CSV 文件）。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
YAML 映射文件示例 (class_map.yaml)
--------------------------------
n01440764: 0
n02102040: 1
n02979186: 2
n03000684: 3
n03028079: 4
n03394916: 5
n03417042: 6
n03425413: 7
n03445777: 8
n03888257: 9

CSV 文件示例 (dataset.csv)
-------------------------
path_img,class_name
imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG,n01440764
imagenette2-320/train/n01440764/ILSVRC2012_val_00002138.JPEG,n01440764
imagenette2-320/train/n01440764/ILSVRC2012_val_00003014.JPEG,n01440764
imagenette2-320/train/n01440764/ILSVRC2012_val_00006697.JPEG,n01440764

运行示例
--------
python add_class_id.py dataset.csv class_map.yaml -c class_name

处理后 CSV 示例
---------------
path_img,class_name,class_id
imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG,n01440764,0
imagenette2-320/train/n01440764/ILSVRC2012_val_00002138.JPEG,n01440764,0
imagenette2-320/train/n01440764/ILSVRC2012_val_00003014.JPEG,n01440764,0
imagenette2-320/train/n01440764/ILSVRC2012_val_00006697.JPEG,n01440764,0
"""
    )

    parser.add_argument(
        "csv_file",
        help="输入 CSV 文件路径，例如 dataset.csv"
    )

    parser.add_argument(
        "yaml_file",
        help="YAML 映射文件路径，例如 class_map.yaml"
    )

    parser.add_argument(
        "-c", "--class-col",
        default="class_name",
        help="CSV 中类别列的列名 (默认: class_name)"
    )

    parser.add_argument(
        "-i", "--id-col",
        default="class_id",
        help="新增的 ID 列名称 (默认: class_id)"
    )

    return parser.parse_args()


def load_yaml_mapping(yaml_path):
    """加载 YAML 映射 {class_name: class_id}"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f)
    except Exception as e:
        print(f"加载 YAML 文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(mapping, dict):
        print("错误：YAML 文件必须是 {class_name: class_id} 的映射", file=sys.stderr)
        sys.exit(1)

    return mapping


def add_class_id(csv_path, mapping, class_col, id_col):
    """为 CSV 添加 class_id 列"""

    tmp_path = csv_path.with_suffix(".tmp")

    total = 0
    mapped = 0
    missing = 0

    with open(csv_path, 'r', encoding='utf-8') as fin, \
         open(tmp_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin)

        if class_col not in reader.fieldnames:
            print(f"错误：CSV 中不存在列 '{class_col}'", file=sys.stderr)
            sys.exit(1)

        fieldnames = reader.fieldnames.copy()

        if id_col not in fieldnames:
            fieldnames.append(id_col)

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1

            class_val = row[class_col].strip()

            if class_val in mapping:
                row[id_col] = mapping[class_val]
                mapped += 1
            else:
                row[id_col] = ""
                missing += 1
                print(f"警告：类别 '{class_val}' 不在 YAML 映射中", file=sys.stderr)

            writer.writerow(row)

    # 用新文件替换原文件
    tmp_path.replace(csv_path)

    print("处理完成")
    print(f"总样本数: {total}")
    print(f"成功映射: {mapped}")
    print(f"未匹配: {missing}")


def main():
    args = parse_args()

    csv_path = Path(args.csv_file)
    yaml_path = Path(args.yaml_file)

    if not csv_path.is_file():
        print(f"错误：CSV 文件 '{csv_path}' 不存在", file=sys.stderr)
        sys.exit(1)

    if not yaml_path.is_file():
        print(f"错误：YAML 文件 '{yaml_path}' 不存在", file=sys.stderr)
        sys.exit(1)

    mapping = load_yaml_mapping(yaml_path)

    add_class_id(
        csv_path=csv_path,
        mapping=mapping,
        class_col=args.class_col,
        id_col=args.id_col
    )


if __name__ == "__main__":
    main()