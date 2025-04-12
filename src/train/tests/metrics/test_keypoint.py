# -*- encoding: utf-8 -*-
"""
@File    :   test_keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/15 00:31:04
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import sys

sys.path.insert(0, "/home/xiaopangdun/project/deep_learning/")
import pytest
import numpy as np
import torch
from src.metrics.keypoint import EuclideanDistance


class TestEuclideanDistance(object):

    @staticmethod
    def test_init():
        # temp: indexs_annotation,indexs_data,indexs_label
        temp_class = EuclideanDistance()
        return

    @staticmethod
    def test_update():
        # temp: pred,target,
        # temp = [
        #     [np.array([[1, 1], [5, 5], [15, 5]]), np.array([[2, 2]])],
        #     [np.array([[1, 1]]), np.array([[2, 2]])],
        #     [np.array([[1, 1]]), np.array([[10, 10]])],
        #     [np.array([[1, 1]]), np.array([[2, 2]])],
        #     [np.array([[1, 1]]), np.array([[2, 2]])],
        #     [np.array([[]]), np.array([[]])],
        #     [np.array([[1, 1]]), np.array([])],
        #     [np.array([]), np.array([[1, 1]])],
        # ]
        # temp_class = EuclideanDistance(
        #     2.0,
        #     5.0,
        #     "xy",
        # )
        # for i, v in enumerate(temp):
        #     pred, target = v[0], v[1]
        #     temp_class.update(pred, target)

        # res = temp_class.compute()
        # assert res["distance_error"] - 0.7071 < 0.01
        # assert res["precision"] - 0.5 < 0.01
        # assert res["recall"] - 0.6666 < 0.01
        # assert res["total"] - 8 < 0.01
        # assert res["correct"] - 4 < 0.01
        # assert res["miss"] - 2 < 0.01
        # assert res["error"] - 4 < 0.01

        # temp: pred,target,
        temp = [
            [np.array([[1, 1, 10], [5, 5, 20], [15, 5, 20]]), np.array([[2, 2, 0],[1,1,10]])],
            [np.array([[1, 1, 10]]), np.array([[2, 2, 10]])],
            [np.array([[2, 1, 10]]), np.array([[10, 10, 10]])],
            [np.array([[3, 1, 10]]), np.array([[2, 2, 3]])],
            [np.array([[4, 1, 10]]), np.array([[2, 2, 6]])],
            [np.array([]), np.array([])],
            [np.array([[1, 1, 10]]), np.array([])],
            [np.array([]), np.array([[1, 1, 10]])],
        ]
        temp_class = EuclideanDistance(
            2.0,
            5.0,
            "xya",
        )
        for i, v in enumerate(temp):
            pred, target = v[0], v[1]
            temp_class.update(pred, target)

        res = temp_class.compute()
        assert res["distance_error"] - 0.7071 < 0.01
        assert res["precision"] - 0.5 < 0.01
        assert res["recall"] - 0.6666 < 0.01
        assert res["total"] - 7 < 0.01
        assert res["correct"] - 3 < 0.01
        assert res["miss"] - 2 < 0.01
        assert res["error"] - 4 < 0.01
        return

    @staticmethod
    def test_demo():
        pass


if __name__ == "__main__":

    TestEuclideanDistance.test_update()
    pass
