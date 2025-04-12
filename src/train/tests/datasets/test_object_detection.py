# -*- encoding: utf-8 -*-
"""
@File    :   test_object_detection.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/03/03 21:55:00
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
import cv2
import numpy as np
from src.datasets.object_detection import DirectionalCornerDetectionDataset

PATH_TXT = [ "../database/ps2.0/val.txt", "../database/ps2.0/test.txt"] # "../database/ps2.0/train.txt",
#HW,CHW
CFGS = {"output_size": [16, 16], "input_size": [3, 512, 512], "classes": ["T","L"]}

class TestDirectionalCornerDetectionDataset:
    # @pytest.fixture
    # 测试正常初始化
    def test_str_initialization(self):
        path_txt = PATH_TXT 
        cfgs = CFGS
        indexs_annotations = ("data_image", "label_0")
        transforms = "train"
        dataset = DirectionalCornerDetectionDataset(path_txt, cfgs, indexs_annotations, transforms=transforms)
        net_in, net_out = dataset[0]

        pass

    def test_draw_on_data(self):
        path_txt = PATH_TXT
        cfgs_dataset = CFGS 
        indexs_annotations = ("data_image", "label_0")
        transforms = "train" # "train", "val", "test" "None"
        dataset = DirectionalCornerDetectionDataset(path_txt, cfgs_dataset, indexs_annotations, transforms=transforms)
        net_in, net_out = dataset[0]
        cfgs=[("L", (0, 255, 0)), ("T", (0, 0, 255))]
        for i in range(len(dataset)):
            net_in, net_out = dataset[i]
            img = net_in["img"]
            img_tensor = net_in["img_tensor"]
            path_img =Path(net_in["img_path"])
            keypoint_valid = net_out["keypoint_valid"]
            keypoint_tensor = net_out["keypoint_tensor"]
            keypoint_valid_new = dataset.convert_tensor_to_valid(keypoint_tensor)
            print(keypoint_valid)
            # img_with_keypoint = dataset.draw_valid_on_data(img, keypoint_valid)
            # img_with_keypoint = dataset.draw_valid_on_data(img, keypoint_valid,path_img=str(path_img))
            # img_with_keypoint = dataset.draw_valid_on_data(img, keypoint_valid_new)
            # img_with_keypoint = dataset.draw_valid_on_data(img, keypoint_valid,keypoint_valid_new)

            # img_new = DirectionalCornerDetectionDataset.convert_image_from_tensor_to_numpy(img_tensor)
            # img_with_keypoint = dataset.draw_valid_on_data(img_new, keypoint_valid)

            img_with_keypoint = dataset.draw_valid_on_data(img, keypoint_valid,{"L":[],"T":[]},cfgs)
            # img_with_keypoint = dataset.draw_tensor_on_data(img_tensor, keypoint_tensor,keypoint_tensor,target_keypoint_type_and_colcor=cfgs)
            cv2.imwrite(f"tmp/{path_img.stem}.jpg", img_with_keypoint)
        pass


    def test_convert_ps20_to_my_json(self):
        DirectionalCornerDetectionDataset.convert_ps20_to_labelme_json("../database/ps2.0")


if __name__ == "__main__":
    temp_class = TestDirectionalCornerDetectionDataset()
    # temp_class.test_str_initialization()
    temp_class.test_draw_on_data()
    # temp_class.test_draw_tensor_on_data()
    # temp_class.test_convert_tensor_to_valid()
    # temp_class.test_convert_ps20_to_my_json()
    pass
