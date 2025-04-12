# -*- encoding: utf-8 -*-
"""
@File    :   test_keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/10/29 22:21:01
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import sys

sys.path.insert(0, "/home/xiaopangdun/project/deep_learning/")
import pytest
import cv2
from src.datasets.keypoint import RegressionPointDataset


class TestRegressionPointDataset(object):

    def test_demo(self):

        path_txt = r"database_sample/parking_slot/train.txt"
        dir_save = r"tmp"
        cfgs = {"threshold":0.5}
        dataset = RegressionPointDataset(path_txt,cfgs)
        for i in range(len(dataset)):
            net_in, net_out = dataset[i]
            img = net_in["img"]
            keypoint_valid = net_out["keypoint_valid"]
            keypoint_tensor = net_out["keypoint_tensor"]
            # img_with_label = dataset.draw_valid_on_data(img, keypoint_valid)
            img_with_label = dataset.draw_tensor_on_data(img,keypoint_tensor)
            cv2.imwrite("{}/{}.jpg".format(dir_save,i), img_with_label)

        pass


# import pytest  #导入pytest模块


# def test_beifan():  #测试用例
#     pass

# class TestBaili:  #测试套件
#     def test_a(self): #测试用例，第一个测试方法
#         pass

#     def test_b(self):  #测试用例，第二个测试方法
#         pass

# assert xx：判断 xx 为真
# assert not xx：判断 xx 不为真
# assert a in b：判断 b 包含 a
# assert a == b：判断 a 等于 b
# assert a !=b：判断 a 不等于 b
if __name__ == "__main__":
    temp_class = TestRegressionPointDataset()
    temp_class.test_demo()
    pass
