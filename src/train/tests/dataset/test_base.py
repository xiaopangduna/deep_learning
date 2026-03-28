from sys import path_hooks
import pytest
import csv
import os
import numpy as np
import shutil
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from lovely_deep_learning.dataset.base import BaseDataset  # 替换为实际模块路径

PATH_TEST_BASE_CSV = Path("tests/test_data/dataset/test_base.csv")

def test_base_dataset():
    dataset = BaseDataset(csv_paths=[PATH_TEST_BASE_CSV,PATH_TEST_BASE_CSV], key_map={"path_img": "path_image"})
    pass

