import os
import csv
import argparse
import random
from typing import List, Tuple

DEFAULT_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')


def list_images_in_class(class_path: str, image_extensions: Tuple[str, ...]) -> List[str]:
    """返回类别文件夹下所有图片的绝对路径"""
    return sorted([
        os.path.join(class_path, fname)
        for fname in os.listdir(class_path)
        if fname.lower().endswith(image_extensions)
    ])


def collect_image_records(data_dir: str, base_dir: str, image_extensions: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """遍历 data_dir 下的类别文件夹，返回 [(相对路径, 类别名), ...]"""
    records = []
    data_dir = os.path.abspath(data_dir)
    base_dir = os.path.abspath(base_dir)

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for file_path in list_images_in_class(class_path, image_extensions):
            rel_path = os.path.relpath(file_path, base_dir)
            records.append((rel_path, class_name))
    return records


def split_records(records: List[Tuple[str, str]], ratios: List[float]) -> List[List[Tuple[str, str]]]:
    """按比例划分记录，返回每个子集"""
    if not records:
        return [[] for _ in ratios]

    total = sum(ratios)
    normalized = [r / total for r in ratios]

    random.shuffle(records)
    n = len(records)
    splits = []
    start = 0
    for ratio in normalized:
        end = start + int(round(ratio * n))
        splits.append(records[start:end])
        start = end

    # 修正最后一个子集，避免丢失记录，但只在多个子集时
    if len(ratios) > 1:
        splits[-1] = records[start:]
    return splits


def write_records_to_files(splits: List[List[Tuple[str, str]]], output_files: List[str]):
    """将每个子集写入对应文件，每行 image_path class_name"""
    if len(splits) != len(output_files):
        raise ValueError("splits 与 output_files 数量不一致")
    for subset, filename in zip(splits, output_files):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("path_image,class\n")
            for path, cls in subset:
                f.write(f"{path},{cls}\n")
        print(f"{filename} 已写入 {len(subset)} 条记录")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为图像分类数据，按比例随机划分图片数据，生成文本文件列表",
        epilog=(
            "目录说明:\n"
            "  data_dir: 数据目录，里面每个类别是一个文件夹，例如:\n"
            "    train/\n"
            "      ├── classA/\n"
            "      │    ├── img1.JPEG\n"
            "      │    ├── img2.JPEG\n"
            "      │    └── ...\n"
            "      ├── classB/\n"
            "      │    ├── img1.JPEG\n"
            "      │    └── ...\n"
            "      └── ...\n"
            "  base_dir: 输出文件中图片路径的相对起点，例如 train/\n"
            "图片后缀可通过 --image_extensions 指定，默认: .jpg,.jpeg,.png,.bmp,.gif\n"
            "示例:\n"
            "  python scripts/generate_classification_dataset_table.py \\\n"
            "      --data_dir /path/to/dir \\\n"
            "      --base_dir /path/to/dir \\\n"
            "      --output_files train.txt,val.txt \\\n"
            "      --split_ratio 0.7,0.3 \\\n"
            "      --image_extensions .jpg,.jpeg,.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help="数据目录，包含各类别子文件夹")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="路径相对起点，用于输出文件列表中的相对路径")
    parser.add_argument('--output_files', type=str, required=True,
                        help="输出文件列表，用逗号分隔，例如 train.txt,val.txt")
    parser.add_argument('--split_ratio', type=str, required=True,
                        help="各文件对应的比例，用逗号分隔，例如 0.7,0.3 或 1.0")
    parser.add_argument('--image_extensions', type=str, default=",".join(DEFAULT_IMAGE_EXTENSIONS),
                        help="图片后缀，用逗号分隔，默认 .jpg,.jpeg,.png,.bmp,.gif")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"data_dir 不存在: {args.data_dir}")
    if not os.path.exists(args.base_dir):
        raise FileNotFoundError(f"base_dir 不存在: {args.base_dir}")

    image_extensions = tuple(ext.strip().lower() for ext in args.image_extensions.split(',') if ext.strip())
    output_files = [f.strip() for f in args.output_files.split(',')]
    ratios = [float(r.strip()) for r in args.split_ratio.split(',')]

    if len(output_files) != len(ratios):
        raise ValueError("output_files 与 split_ratio 数量必须一致")

    records = collect_image_records(args.data_dir, args.base_dir, image_extensions)
    print(f"共收集 {len(records)} 条图片记录")

    splits = split_records(records, ratios)
    write_records_to_files(splits, output_files)


if __name__ == "__main__":
    main()