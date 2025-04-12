# -*- encoding: utf-8 -*-
"""
@File    :   get_txt_for_dataset.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/1/13 23:33:13
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   目的是自动化生成用于模型训练的数据集，并确保数据集的正确分割和保存。
"""
import os
import sys

path_parent_dir = os.path.dirname(sys.path[0])
sys.path.insert(0, path_parent_dir)

import argparse
from pathlib import Path
from typing import List
import random
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatasetGenerator:
    """
    A utility class for generating datasets for model training.
    Combines paths of input and output data from multiple folders into a list.
    Each element corresponds to paths of input and output data, formatted as: input1, input2, ..., output1, output2, ...
    """

    @staticmethod
    def generate_dataset(dirs: List[str], suffixs: List[List[str]],relative_path=None) -> List[str]:
        """Generate a list containing paired input and output file paths.

        Args:
            dirs (List[str]): A list of directories for input or output data.
            suffixs (List[List[str]]): A list of lists, where each inner list contains file suffixes for the corresponding directory.

        Returns:
            List[str]: A list of strings, where each string is a comma-separated line of file paths.

        Raises:
            ValueError: If the number of directories does not match the number of suffix groups.
            FileNotFoundError: If any directory does not exist.
        """
# 检查并记录重复的文件名主体
        def record_duplicate_bases(file_lists):
            duplicate_records = []
            for idx, file_list in enumerate(file_lists):
                base_counts = defaultdict(list)
                for file in file_list:
                    base, ext = os.path.splitext(file)
                    base_counts[base].append(file)
                duplicates = {base: files for base, files in base_counts.items() if len(files) > 1}
                if duplicates:
                    duplicate_records.append((idx + 1, duplicates))
            return duplicate_records

        # 提取每个列表中的文件名主体和后缀
        def extract_bases_and_extensions(file_lists):
            bases_and_extensions = []
            for file_list in file_lists:
                bases = []
                extensions = []
                for file in file_list:
                    base, ext = os.path.splitext(file)
                    bases.append(base)
                    extensions.append(ext)
                bases_and_extensions.append((bases, extensions))
            return bases_and_extensions

        # 找出所有列表的共同文件名主体
        def find_common_bases(bases_list):
            if not bases_list:
                return set()
            common_bases = set(bases_list[0])
            for bases in bases_list[1:]:
                common_bases &= set(bases)
            return common_bases

        # 将共同文件名主体和后缀拼回去
        def restore_full_filenames(common_bases, bases_and_extensions):
            restored_files = []
            for bases, extensions in bases_and_extensions:
                files = []
                for base, ext in zip(bases, extensions):
                    if base in common_bases:
                        files.append(base + ext)
                restored_files.append(files)
            return restored_files

        # 主函数
        def process_file_lists(file_lists):
            # 记录重复的文件名主体
            duplicate_records = record_duplicate_bases(file_lists)
            
            # 打印重复记录
            for list_idx, duplicates in duplicate_records:
                print(f"Duplicate bases in list {list_idx}:")
                for base, files in duplicates.items():
                    print(f"  Base '{base}' appears in files: {files}")
            
            # 提取文件名主体和后缀
            bases_and_extensions = extract_bases_and_extensions(file_lists)
            
            # 找出共同文件名主体
            common_bases = find_common_bases([bases for bases, _ in bases_and_extensions])
            
            # 拼回完整的文件名
            restored_files = restore_full_filenames(common_bases, bases_and_extensions)
            
            return restored_files, common_bases




        # Check if the number of directories matches the number of suffix groups
        if len(dirs) != len(suffixs):
            raise ValueError("The number of directories must match the number of suffix groups")

        # Check if each directory exists
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory does not exist: {dir_path}")

        # List to store path combinations for each line
        dataset_lines = []


        file_groups = [
            sorted([f for f in os.listdir(dir_path) if any(f.endswith(suffix) for suffix in suffix_group)])
            for dir_path, suffix_group in zip(dirs, suffixs)
        ]
        # 示例使用
        file_groups, common_bases = process_file_lists(file_groups)


        for i in range(1, len(file_groups)):
            if len(file_groups[i]) != len(file_groups[0]):
                raise ValueError(f"The number of files in folder {dirs[i]} does not match {dirs[0]}")
        # Combine the paths of files in each group into a line
        for file_index in range(len(file_groups[0])):
            # Get the paths of files
            file_paths = [os.path.join(dir_path, file_groups[i][file_index]) for i, dir_path in enumerate(dirs)]
            # Combine the paths into a line, separated by commas
            for i,file_path in enumerate(file_paths):
                if relative_path:
                    file_path = os.path.relpath(file_path, relative_path)
                file_paths[i] = file_path
            line = " ".join(file_paths)
            dataset_lines.append(line)

        return dataset_lines


    @staticmethod
    def split_dataset(dataset: List[str], split_ratio: List[float]) -> List[List[str]]:
        """
        Dynamically split the dataset according to split_ratio.

        Args:
            dataset (List[str]): The dataset list.
            split_ratio (List[float]): The split ratio list, the sum must be 1.0.

        Returns:
            List[List[str]]: The list of split datasets, each element is a subset.
        """
        # Check if the sum of split_ratio is 1.0
        if not sum(split_ratio) == 1.0:
            raise ValueError("The sum of split_ratio must be 1.0")

        # Calculate the start and end indices for each subset
        total = len(dataset)
        subsets = []
        start = 0
        for ratio in split_ratio:
            end = start + int(total * ratio)
            subsets.append(dataset[start:end])
            start = end

        return subsets

    @staticmethod
    def save_datasets_to_files(datasets, filenames, dir_save):
        dir_save = Path(dir_save)
        dir_save.mkdir(parents=True, exist_ok=True)
        for dataset, filename in zip(datasets, filenames):
            with dir_save.joinpath(filename).open("w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(item + "\n")


def validate_split_ratio(split_ratio):
    """Validate the split-ratio parameter"""

    ratios = [float(ratio) for ratio in split_ratio]
    if sum(ratios) != 1.0:
        raise ValueError("The sum of split-ratio values must be 1.0")
    return ratios


def main():
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Generate dataset files for model training.")
    parser.add_argument(
        "--dirs-group",
        nargs="+",
        default=["/home/xiaopangdun/project/database/ps2.0/training,/home/xiaopangdun/project/database/ps2.0/labelme_json/training"],
        help="List of folder groups, each group separated by commas, the whole separated by spaces. For example: 'dir1,dir2' 'dir3,dir4'",
    )
    parser.add_argument(
        "--suffixs",
        nargs="+",
        default=[".jpg,.png", ".json"],
        help="List of suffix groups, each group separated by commas, the whole separated by spaces. For example: '.jpg,.png' '.json' '.txt'",
    )
    parser.add_argument("--dir-save", default="/home/xiaopangdun/project/database/ps2.0", help="Directory to save output files")
    parser.add_argument(
        "--split-ratio",
        nargs="+",
        default=[0.75,0.25],
        help="Split ratio for train, val, and test. For example: 0.6 0.2 0.2",
    )
    parser.add_argument(
        "--output-names",
        nargs="+",
        default=["train.txt","val.txt"],
        help="List of output file names, corresponding one-to-one with split-ratio. For example: 'train val test'",
    )
    parser.add_argument("--shuffle", type=bool, default=False, help="shuffle for dataset splitting")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--relative",
        type=bool,
        default=True,
        help="",
    )
    # Parse command line arguments
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Validate split-ratio parameter
    try:
        split_ratio = validate_split_ratio(args.split_ratio)
    except ValueError as e:
        logging.error(f"Parameter error: {e}")
        return

    # Check if the lengths of output-names and split-ratio are consistent
    if len(args.output_names) != len(split_ratio):
        logging.error("The lengths of output-names and split-ratio must be consistent")
        return
    # Process suffixs parameter
    suffixs = [suffix_group.split(",") for suffix_group in args.suffixs]

    # Generate dataset
    try:
        dataset = []
        dirs_group = [dirs.split(",") for dirs in args.dirs_group]
        if args.relative:
            relative_path = args.dir_save
        else:
            relative_path = None
        for dirs in dirs_group:
            dataset += DatasetGenerator.generate_dataset(dirs, suffixs,relative_path)
    except ValueError as e:
        logging.error(f"Error: {e}")
        return
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return

    # Shuffle randomly
    if args.shuffle:
        random.shuffle(dataset)

    # Split dataset
    subsets = DatasetGenerator.split_dataset(
        dataset,
        split_ratio,
    )
    # Save results
    DatasetGenerator.save_datasets_to_files(subsets, args.output_names, args.dir_save)
    # Log output
    logging.info(f"Total number of datasets: {len(dataset)}")
    for name, data in zip(args.output_names, subsets):
        logging.info(f"Number of {name} set: {len(data)}")
    logging.info(f"Datasets have been generated and saved to: {str(args.dir_save)}")


if __name__ == "__main__":
    main()
