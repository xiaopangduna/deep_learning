# -*- encoding: utf-8 -*-
"""
@File    :   test_trainer_keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/13 22:51:47
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import sys

sys.path.insert(0, "/home/xiaopangdun/project/deep_learning/")
import pytest

from src.datasets.object_detection import DirectionalCornerDetectionDataset


class TestBaseDataset(object):
    def test_get_model(self):
        model_name = "DMPR"
        model_hparams = {"input_size": (1, 3, 512, 512), "feature_map_channel": 6, "depth_factor": 32}
        optimizer_name = "Adam"
        optimizer_hparams = {"lr": 1e-3, "weight_decay": 1e-4}
        module = KeypointModule(model_name, model_hparams, optimizer_name, optimizer_hparams)

        pass


if __name__ == "__main__":
    temp_class = TestBaseDataset()
    temp_class.test_get_model()
