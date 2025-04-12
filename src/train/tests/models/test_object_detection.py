import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
import cv2
import torch
import numpy as np
from src.models.object_detection import DirectionalCornerDetectionModel

class TestDirectionalCornerDetectionModel:
    # @pytest.fixture
    # 测试正常初始化
    def test_model_initialization(self):
        cfgs = {"backbone_name" : "resnet18", "pretrained" : False ,"input_size": [1, 3, 512, 512],  "output_size": [1,7,16, 16]}
        model = DirectionalCornerDetectionModel(**cfgs)
        input_tensor = torch.randn(1, 3, 512, 512)
        # 前向传播，获取预测结果
        output = model(input_tensor)
        print("Output shape:", output.shape)
        pass

if __name__ == "__main__":
    temp_class = TestDirectionalCornerDetectionModel()
    # temp_class.test_str_initialization()
    temp_class.test_model_initialization()
    # temp_class.test_draw_tensor_on_data()
    # temp_class.test_convert_tensor_to_valid()
    # temp_class.test_convert_ps20_to_my_json()
    pass
