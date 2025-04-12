# -*- encoding: utf-8 -*-
"""
@File    :   test_base_dataset.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/11 23:10:22
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from pathlib import Path
from src.datasets.base_dataset import BaseDataset


class TestBaseDataset:
    # @pytest.fixture
    # 测试正常情况
    # 测试列表作为输入
    def test_list_initialization(self):
        path_list = ["database_sample/parking_slot/train.txt", "database_sample/parking_slot/train.txt"]
        cfgs = {"param2": "value2"}
        indexs_annotations = ("data_image", "label_1")
        dataset = BaseDataset(path_list, cfgs, indexs_annotations)
        assert dataset.cfgs == cfgs
        assert dataset.indexs_annotations == indexs_annotations

    # 测试异常情况
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            BaseDataset(123)  # 输入不是字符串或列表

        with pytest.raises(ValueError):
            BaseDataset("", indexs_annotations=("invalid_index", "label_0"))  # 无效的索引

        with pytest.raises(FileNotFoundError):
            BaseDataset("nonexistent.txt")  # 文件不存在


if __name__ == "__main__":
    temp_class = TestBaseDataset()
    temp_class.test_list_initialization()
    temp_class.test_invalid_input()

    pass
