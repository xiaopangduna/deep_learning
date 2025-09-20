# -*- coding: utf-8 -*-
"""
@File    :   file_processor.py
@Time    :   2024/03/20 22:15:52
@Author  :   xiaopangdun
@Email  :   18675381281@163.com
@Version :   1.0
@Desc    :   None
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Set, Any, Iterable
from pathlib import Path
import random

def list_grouped_files_from_folders(
    dirs: List[Union[str, Path]],
    suffix_groups: List[List[str]],
    relative_to: Optional[Union[str, Path]] = None,
    allow_missing: bool = True,
    verbose: bool = False,  # Controls whether to print detailed statistics
) -> List[List[str]]:
    """
    Lists grouped files from multiple folders, aligned by their base filenames.

    Each row represents a group of files from different folders:
    - If allow_missing=True, missing files are replaced with "None";
    - If allow_missing=False, rows with any missing file are discarded;
    - If a folder contains files with the same stem but different suffixes,
      a warning is issued and the entire group is discarded.

    Args:
        dirs: List of folder paths, each folder corresponds to a column
        suffix_groups: List of allowed file suffixes for each folder, must have the same length as dirs
        relative_to: Optional path, if provided, returned paths will be relative to this directory
        allow_missing: Whether to allow missing files, default is True
        verbose: Whether to print detailed processing statistics, default is False

    Returns:
        2D list of file paths, each row is a group of aligned files
    """
    if len(dirs) != len(suffix_groups):
        raise ValueError("dirs and suffix_groups must have the same length")

    relative_path = Path(relative_to) if relative_to else None
    file_maps = []
    # Stores duplicate stems for each folder: {folder_path: {duplicate_stems}}
    folder_duplicates: Dict[str, Set[str]] = {}
    duplicate_stems = set()  # Global set of duplicate stems
    folder_raw_counts = []  # Records number of valid files for each folder

    for folder_path, allowed_suffixes in zip(dirs, suffix_groups):
        folder = Path(folder_path)
        folder_str = str(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {folder}")

        # 1. Collect valid files and their stems
        valid_files = []
        for file in folder.iterdir():
            if file.suffix in allowed_suffixes:
                valid_files.append((file.stem, file))
        folder_raw_counts.append(len(valid_files))

        # 2. Check for duplicate stems in current folder
        stem_tracker: Dict[str, int] = {}  # Tracks count of each stem
        for stem, _ in valid_files:
            stem_tracker[stem] = stem_tracker.get(stem, 0) + 1

        # Identify stems that appear more than once
        current_duplicates = {stem for stem, count in stem_tracker.items() if count > 1}
        if current_duplicates:
            folder_duplicates[folder_str] = current_duplicates
            duplicate_stems.update(current_duplicates)

        # 3. Build file map (excluding duplicate stems)
        file_map = {}
        for stem, file in valid_files:
            if stem not in current_duplicates:
                file_path = str(file.relative_to(relative_path)) if relative_path else str(file)
                file_map[stem] = file_path
        file_maps.append(file_map)

    # Calculate valid base names
    if allow_missing:
        all_bases = set().union(*file_maps)
        candidate_count = len(all_bases)
    else:
        all_bases = set.intersection(*map(set, file_maps)) if file_maps else set()
        candidate_count = len(all_bases)
    valid_bases = all_bases - duplicate_stems
    final_count = len(valid_bases)

    # Print detailed information if verbose mode is enabled
    if verbose:
        print("=" * 80)
        print("File Grouping Processing Details")
        print("=" * 80)

        # 1. Input folder statistics
        for i, (folder, count) in enumerate(zip(dirs, folder_raw_counts)):
            print(f"Folder {i+1} [{folder}]: Number of valid files = {count}")

        # 2. Duplicate file information
        if folder_duplicates:
            print("\nFound files with duplicate stems (same name, different suffixes):")
            for folder, stems in folder_duplicates.items():
                print(f"  Folder: {folder}")
                for stem in sorted(stems):
                    print(f"    Duplicate stem: {stem}")
            print(f"\nTotal of {len(duplicate_stems)} duplicate stems were filtered out")
        else:
            print("\nNo files with duplicate stems found")

        # 3. File group statistics
        print("\nFile group statistics:")
        print(f"  Total candidate groups (before filtering): {candidate_count}")
        print(f"  Final valid groups: {final_count}")

        # 4. Filtering ratio
        if candidate_count > 0:
            filtered_ratio = (candidate_count - final_count) / candidate_count * 100
            print(f"  Filtering ratio: {filtered_ratio:.1f}%")

        print("=" * 80)

    # Generate final result
    return [[file_map.get(base, "") for file_map in file_maps] for base in sorted(valid_bases)]

def split_list_by_ratio(
    items: Iterable[Any],  # 1. 修正参数名：从grouped_files→items（通用化）
    ratios: List[float],
    shuffle: bool = True  # 2. 增加默认值：简化无特殊需求的调用
) -> List[List[Any]]:  # 3. 修正返回值类型：原List[List[List[Any]]]是过度嵌套
    """
    按比例将可迭代对象拆分为多个子列表。
    
    特性：
    - 自动处理比例总和与实际数量的匹配（最后一组分配剩余元素）
    - 支持拆分前打乱顺序
    - 输入可迭代对象（列表、元组等），输出统一为列表
    
    参数：
        items: 待拆分的可迭代对象（如列表、元组、文件组列表等）
        ratios: 拆分比例列表（需为正数，总和应接近1.0，如 [0.7, 0.2, 0.1]）
        shuffle: 拆分前是否打乱 items 的顺序，默认 True
    
    返回：
        List[List[Any]]: 按比例拆分后的子列表组成的列表，长度与 ratios 一致
        例：拆分 [1,2,3,4,5] 为 [0.6,0.4]，可能返回 [[3,1,5], [2,4]]
    
    异常：
        ValueError: 若 ratios 包含非正数
        TypeError: 若 items 不可迭代或 ratios 非列表
    """
    # 4. 确保输入可迭代且转换为列表（适配元组等其他可迭代类型）
    try:
        items_list = list(items)
    except TypeError as e:
        raise TypeError(f"items 必须是可迭代对象（如列表、元组），当前类型：{type(items)}") from e
    
    total = len(items_list)
    if total == 0:
        return [[] for _ in ratios]
    
    # 5. 加强参数校验（公共函数必备：提前暴露错误）
    if not isinstance(ratios, list) or not all(isinstance(r, (int, float)) for r in ratios):
        raise TypeError("ratios 必须是浮点数/整数组成的列表（如 [0.7, 0.3]）")
    
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError(f"拆分比例必须为正数，当前存在无效值：{ratio}")
    
    # 打乱顺序（核心逻辑不变）
    if shuffle:
        items_list = random.sample(items_list, k=total)
    
    # 计算每组数量（核心逻辑不变，仅修正变量名适配通用场景）
    counts = []
    remaining = total
    for ratio in ratios[:-1]:
        count = int(round(total * ratio))
        count = min(count, remaining)  # 避免比例计算溢出
        counts.append(count)
        remaining -= count
    counts.append(remaining)  # 最后一组兜底剩余元素
    
    # 切片拆分（核心逻辑不变）
    split_result = []
    start = 0
    for count in counts:
        end = start + count
        split_result.append(items_list[start:end])
        start = end
    
    return split_result

class FileProcessor(object):

    @staticmethod
    def rename_file_order(
        dir_input: str,
        dir_output: str = None,
        initial_num: int = 1,
        prefix: str = "",
    ):
        """Copy and rename files.

        Args:
            dir_input (str): The path of floder which save original file.
            dir_output (str): The path of floder which save target file.
            initial_num (int, optional): The initial number of the file name. Defaults to 1.
            prefix (str, optional): The prefix of file name . Defaults to "".
            separator (str, optional): The separator of file name. Defaults to "_".
            suffix (str, optional): THe suffix of file name. Defaults to "".

        Example:
            path_input = r"D:/A_Project/database/park_slot/train_harbor_vital"
            path_output = r"D:/A_Project/database/park_slot/train_harbor_vital"
            initial_num = 1
            prefix = "240316"
            separator = "_"
            suffix = "03"
            processor = FileProcessor()
            processor.copy_and_rename_files(
                path_input, path_output, initial_num, prefix, separator, suffix
            )
            # file name
            # 240316_03_00001.jpg
        """
        # check dir is exits.
        if not os.path.isdir(dir_input):
            print("Error  :floder is not exit.")
            print("path of dir_input: {}".format(dir_input))
        # get file name
        names = os.listdir(dir_input)
        # names.sort(key=key)
        for name in names:
            # copy and renmae file
            path_old = os.path.join(dir_input, name)
            if dir_output and dir_input != dir_output:
                path_new = os.path.join(
                    dir_output,
                    "{}{}{}{}{:05d}{}".format(
                        prefix,
                        separator,
                        suffix,
                        separator,
                        initial_num,
                        os.path.splitext(name)[-1],
                    ),
                )
                shutil.copyfile(path_old, path_new)
                print("Success :copy {} to {}".format(path_old, path_new))
            else:
                path_new = os.path.join(
                    dir_input,
                    "{}{:05d}{}".format(
                        prefix,
                        initial_num,
                        os.path.splitext(name)[-1],
                    ),
                )
                os.rename(path_old, path_new)
                print("Success :rename {} to {}".format(path_old, path_new))
            initial_num += 1

    @staticmethod
    def rename_file(
        dir_input: str,
        dir_output: str = None,
        prefix: str = "",
    ):
        """Copy and rename files.

        Args:
            dir_input (str): The path of floder which save original file.
            dir_output (str): The path of floder which save target file.
            initial_num (int, optional): The initial number of the file name. Defaults to 1.
            prefix (str, optional): The prefix of file name . Defaults to "".
            separator (str, optional): The separator of file name. Defaults to "_".
            suffix (str, optional): THe suffix of file name. Defaults to "".

        Example:
            path_input = r"D:/A_Project/database/park_slot/train_harbor_vital"
            path_output = r"D:/A_Project/database/park_slot/train_harbor_vital"
            initial_num = 1
            prefix = "240316"
            separator = "_"
            suffix = "03"
            processor = FileProcessor()
            processor.copy_and_rename_files(
                path_input, path_output, initial_num, prefix, separator, suffix
            )
            # file name
            # 240316_03_00001.jpg
        """
        # check dir is exits.
        if not os.path.isdir(dir_input):
            print("Error  :floder is not exit.")
            print("path of dir_input: {}".format(dir_input))
        # get file name
        names = os.listdir(dir_input)
        for name in names:
            # copy and renmae file
            path_old = os.path.join(dir_input, name)
            path_new = os.path.join(
                dir_output,
                "{}{}".format(
                    prefix,
                    name,
                ),
            )
            os.rename(path_old, path_new)
            print("Success :rename {} to {}".format(path_old, path_new))


if __name__ == "__main__":

    path_input = r"D:\A_Project\database\bdd100k"
    path_output = r"D:\A_Project\database\bdd100k"
    initial_num = 1
    prefix = "BDD100k_"
    processor = FileProcessor()
    processor.rename_file_order(
        path_input,
        path_output,
        initial_num,
        prefix,
    )
    # processor = FileProcessor()
    # path_aaaa = Path(r"D:\A_Project\database\temp_highway")
    # for path in path_aaaa.iterdir():
    #     path_input = path
    #     path_output = path_input
    #     prefix = "CULane_{}.MP4_".format(path_input.name[:-4])
    #     processor.rename_file(
    #         path_input, path_output,prefix
    #     )
    pass
