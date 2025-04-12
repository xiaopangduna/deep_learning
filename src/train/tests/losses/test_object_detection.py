import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.losses.object_detection import DirectionalCornerDetectionLoss

class TestDirectionalCornerDetectionLoss(object):
    def test_loss(self):
        # 假设 batch_size=2, grid_size=13, num_classes=80
        batch_size = 2
        grid_size = 16
        num_classes = 2

        # 生成随机的预测值和目标值
        pred = torch.randn(batch_size, 5 + num_classes,grid_size, grid_size, )
        targets = torch.randn(batch_size, 5 + num_classes, grid_size, grid_size,)

        # 确保类别概率部分是有效的概率分布
        targets[..., 5:] = F.softmax(targets[..., 5:], dim=-1)

        # 初始化损失函数
        loss_fn = DirectionalCornerDetectionLoss(num_classes=num_classes)

        # 计算损失
        loss = loss_fn(pred, targets)

        print(f"总损失: {loss.item()}")

if __name__ == "__main__":
    temp_class = TestDirectionalCornerDetectionLoss()
    temp_class.test_loss()
