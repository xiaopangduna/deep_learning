import pytest
from pathlib import Path

from lovely_deep_learning.dataset.base import BaseDataset

PATH_TEST_BASE_CSV = Path("tests/test_data/dataset/test_base.csv")


def test_base_dataset():
    dataset = BaseDataset(csv_paths=[PATH_TEST_BASE_CSV, PATH_TEST_BASE_CSV], key_map={
                          "path_img": "path_image"})
    assert len(dataset.sample_path_table) == 4
