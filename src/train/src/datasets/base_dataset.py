# -*- encoding: utf-8 -*-
"""
@File    :   base_dataset.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/11 22:01:17
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import warnings
from typing import Tuple, List, Union
from pathlib import Path

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        path_txts: list,
        cfgs: dict = {},
        indexs_annotations: Tuple[str, str] = ("data_image", "label_0"),
        transforms: str = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            path_txt_or_list (Union[str, Path, list]): The path of a txt file or a list of paths.
                If it's a txt file, each line contains paths of data and labels.
                If it's a list, each element can be a txt file path (str or Path) or a direct data path.
            cfgs (dict, optional): Configuration parameters. Defaults to {}.
            indexs_annotations (tuple, optional): Indexes for data and label in each line. Defaults to ("data_image", "label_0").
            transforms (str, optional): Indicates the type of data transforms (train, val, test, or None). Defaults to None.
        """
        self.cfgs = cfgs
        self.indexs_annotations = indexs_annotations
        self.transforms = self._get_transforms(transforms)
        self.path_txts = path_txts

        # Parse paths from input
        self.path_datas, self.path_labels = self._parse_paths(path_txts)

    def _get_transforms(self, transforms: str):
        """
        根据传入的转换类型字符串获取相应的数据转换函数。

        Args:
            transforms (str): 指示数据转换的类型，可选值为 "train", "val", "test" 或其他值。

        Returns:
            Union[Callable, None]: 如果转换类型为 "train"，返回训练时的数据转换函数；
                                   如果转换类型为 "val" 或 "test"，返回验证/测试时的数据转换函数；
                                   否则返回 None。
        """
        # 如果转换类型为 "train"，调用训练时的数据转换函数
        if transforms == "train":
            return self.get_transforms_for_train()
        # 如果转换类型为 "val" 或 "test"，调用验证/测试时的数据转换函数
        elif transforms in ["val", "test"]:
            return self.get_transforms_for_val()
        # 如果转换类型不是上述三种情况，返回 None
        else:
            return None

    def _parse_paths(self, path_txts: list):
        """
        从输入中解析数据和标签的路径。

        Args:
            path_txt_or_list (Union[str, Path, list]): 一个文本文件的路径或者路径列表。
                如果是文本文件路径，每一行包含数据和标签的路径。
                如果是列表，每个元素可以是文本文件路径（字符串或 Path 对象）或者直接的数据路径。

        Returns:
            Tuple[List[List[str]], List[List[str]]]: 解析后的数据路径列表和标签路径列表。
        """
        # 初始化数据路径列表
        path_datas = []
        # 初始化标签路径列表
        path_labels = []
        # 从 indexs_annotations 中提取数据和标签的索引
        self.indexs_data, self.indexs_label = self.get_indexs_data_and_indexs_label_from_indexs_annotation(
            self.indexs_annotations
        )
        # 遍历列表中的每个元素
        for path_txt in path_txts:
            # 检查元素是否为文本文件
            if os.path.isfile(path_txt) and path_txt.endswith(".txt"):
                # 从文本文件中读取每一行
                with open(path_txt, "r") as f:
                    lines = f.readlines()
                # 解析文件中的每一行
                self._parse_lines(lines, path_datas, path_labels,path_txt)


        return path_datas, path_labels

    def _parse_lines(self, lines: List[str], path_datas: List[List[str]], path_labels: List[List[str]],path_txt):
        """
        解析文本文件中的每一行，提取数据和标签的路径。

        Args:
            lines (List[str]): 从文本文件中读取的行列表。
            path_datas (List[List[str]]): 用于存储解析后的数据路径列表。
            path_labels (List[List[str]]): 用于存储解析后标签路径列表。
        """
        # 遍历每一行
        for line in lines:
            # 去除行首尾的空白字符
            line = line.strip()
            # 如果行为空，则跳过当前循环
            if not line:
                continue

            # 按空格分割每一行
            parts = line.split(" ")
            # 检查分割后的部分数量是否小于索引注释的数量
            if len(parts) < len(self.indexs_annotations):
                # 若小于，发出警告并跳过当前行
                warnings.warn(f"Invalid line format: {line}. Ignored.")
                continue

            # 临时存储当前行的数据路径
            tmp_data = []
            # 临时存储当前行的标签路径
            tmp_label = []
            # 遍历数据索引
            for i in self.indexs_data:
                # 检查索引是否在分割后的部分范围内
                if i < len(parts):
                    # 若在范围内，将对应部分添加到临时数据列表
                    tmp_path = Path(parts[i])
                    if not tmp_path.is_absolute():
                        tmp_path = Path(path_txt).parent.resolve() / tmp_path
                    tmp_data.append(str(tmp_path))
                else:
                    # 若不在范围内，发出索引超出范围的警告
                    warnings.warn(f"Index {i} out of range for line: {line}")
            # 遍历标签索引
            for i in self.indexs_label:
                # 检查索引是否在分割后的部分范围内
                if i < len(parts):
                    # 若在范围内，将对应部分添加到临时标签列表
                    tmp_path = Path(parts[i])
                    if not tmp_path.is_absolute():
                        tmp_path = Path(path_txt).parent.resolve() / tmp_path
                    tmp_label.append(str(tmp_path))
                else:
                    # 若不在范围内，发出索引超出范围的警告
                    warnings.warn(f"Index {i} out of range for line: {line}")

            # 将临时数据列表添加到最终的数据路径列表
            path_datas.append(tmp_data)
            # 将临时标签列表添加到最终的标签路径列表
            path_labels.append(tmp_label)

    def get_indexs_data_and_indexs_label_from_indexs_annotation(
        self, indexs_annotation: Tuple[str, str]
    ) -> Tuple[List[int], List[int]]:
        """
        根据给定的注释信息提取数据和标签的索引。

        Args:
            indexs_annotation (Tuple[str, str]): 一个包含注释信息的元组，每个元素应为以 'data' 或 'label' 开头的字符串。

        Returns:
            Tuple[List[int], List[int]]: 一个元组，包含两个列表，分别为数据的索引列表和标签的索引列表。
        """
        # 初始化数据索引列表
        indexs_data = []
        # 初始化标签索引列表
        indexs_label = []
        # 遍历注释信息元组，同时获取索引和对应的值
        for i, v in enumerate(indexs_annotation):
            # 如果值以 'data' 开头
            if v.startswith("data"):
                # 将当前索引添加到数据索引列表
                indexs_data.append(i)
            # 如果值以 'label' 开头
            elif v.startswith("label"):
                # 将当前索引添加到标签索引列表
                indexs_label.append(i)
            # 如果值既不以 'data' 也不以 'label' 开头
            else:
                # 发出警告，提示该值将被忽略
                warnings.warn(
                    f"{v} not start with 'data' or 'label'. It will be ignored in indexs_annotation "
                    f"and not used during training."
                )
        # 如果数据索引列表为空
        if not indexs_data:
            # 抛出异常，提示未找到 'data' 项
            raise ValueError("No 'data' items found in indexs_annotation")
        # 返回数据索引列表和标签索引列表组成的元组
        return indexs_data, indexs_label

    def __len__(self):
        return len(self.path_datas)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def get_transforms_for_train():
        """Set up the data enhancer during training."""
        pass

    def get_transforms_for_val():
        """Set up the data enhancer for validation/testing."""
        pass

    def save_config(self):
        pass

    @staticmethod
    def filter_shapes_from_labelme(labelme: dict, shapes: dict):
        """Extract data from a dictionary formatted from labelme according to label.

        Args:
            labelme (dict): A dict in labelme format
            shapes (dict): A dictionary whose key is label and whose value is an empty list,such as
                {"X_box":[],"U_box":[]}

        Returns:
            shapes (dict): A dictionary contains the extracted data.

        Example:
            shapes = {"X_box":[],"U_box":[]}
            shapes = filter_shapes_from_labelme(shapes)
        """
        for shape in labelme["shapes"]:
            label = shape["label"]
            shapes[label].append(shape["points"])
        return shapes

    @staticmethod
    def convert_raw_to_valid(self):
        """Extract valid data from raw data."""
        pass

    @staticmethod
    def convert_valid_to_tensor(self):
        pass

    @staticmethod
    def convert_tensor_to_valid(self):
        pass

    @staticmethod
    def convert_valid_to_raw(self):
        pass

    @staticmethod
    def draw_valid_on_data(self, image, true_valid, perdict_valid):
        """
        该方法用于在数据上绘制有效的标注信息。

        此方法通常会接收有效数据（如关键点、边界框等），并将这些信息可视化地绘制在原始数据（如图像）上。
        绘制的目的可能是为了数据检查、调试模型或者展示标注结果。

        目前该方法为空，后续可以根据具体需求实现绘制逻辑，例如使用 OpenCV 或其他图像处理库进行绘制操作。

        Returns:
            None
        """
        pass

    @staticmethod
    def draw_tensor_on_data(self, image, true_tensor, perdict_tensor):
        pass

    @staticmethod
    def draw_valid_on_data_as_grid(images, true_valid, predict_valid):
        pass

    @staticmethod
    def draw_tensor_on_data_as_grid(images, true_tensor, predict_tensor):
        pass

    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(x):
            return list(zip(*x))
        return collate_fn
