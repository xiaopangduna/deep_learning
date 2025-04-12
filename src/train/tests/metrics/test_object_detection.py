import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.metrics.object_detection import DirectionalCornerDetectionMetric
from src.datasets.object_detection import DirectionalCornerDetectionDataset


def get_test_data():
    # 生成测试数据

    # 第一组数据
    data1 = {"L": [], "T": [(123.45, 67.89, 45.67), (234.56, 78.90, 123.45), (345.67, 89.01, 234.56)]}

    # 第二组数据
    data2 = {
        "L": [(456.78, 101.23, 345.67), (511.89, 112.34, 256.78)],  # 角度归一化到0-360
        "T": [(123.45, 67.89, 45.67), (234.56, 78.90, 123.45), (290.12, 145.67, 289.01), (101.23, 156.78, 190.12)],
    }

    # # 第三组数据
    # data3 = {
    #     'L': [],
    #     'T': []
    # }

    # # 第四组数据
    # data4 = {
    #     'L': [
    #         (101.23, 202.34, 56.78),
    #         (202.34, 303.45, 123.45),
    #         (303.45, 404.56, 234.56),
    #         (404.56, 505.67, 345.67)
    #     ],
    #     'T': [
    #         (505.67, 606.78, 456.78 % 360),
    #         (606.78, 707.89, 567.89 % 360)
    #     ]
    # }

    # # 第五组数据
    # data5 = {
    #     'L': [
    #         (707.89, 808.00, 678.90 % 360)
    #     ],
    #     'T': [
    #         (808.00, 909.11, 789.01 % 360),
    #         (909.11, 1010.22, 890.12 % 360),
    #         (1010.22, 1111.33, 901.23 % 360)
    #     ]
    # }
    # 生成测试数据
    return [data1], [data2]


def convert_valid_to_tensor(valid):
    # 假设 valid 是一个包含多个字典的列表
    # 每个字典包含 'L' 和 'T' 两个键，对应的值是一个包含多个元组的列表
    # 初始化一个空的张量列表
    batch_size = len(valid)
    res = []
    for i in range(batch_size):
        data = valid[i]
        # 初始化一个全零的张量，形状为 (5 + num_classes, grid_size, grid_size)
        tensor = DirectionalCornerDetectionDataset.convert_valid_to_tensor(data)
        res.append(tensor)
    res = torch.stack(res, 0)
    return res


if __name__ == "__main__":

    # 假设 batch_size=2, num_classes=80, grid_size=13
    batch_size = 2
    num_classes = 2
    grid_size = 16

    # 生成随机的预测值和目标值
    # pred = torch.randn(batch_size, 5 + num_classes, grid_size, grid_size)
    # targets = torch.randn(batch_size, 5 + num_classes, grid_size, grid_size)
    target_valids,pred_valids = get_test_data()
    target_tensors = convert_valid_to_tensor(target_valids)
    pred_tensors = convert_valid_to_tensor(pred_valids)

    targets = target_tensors
    pred = pred_tensors 
    # 确保类别概率部分是有效的概率分布
    # targets[:, 5:, :, :] = F.softmax(targets[:, 5:, :, :], dim=1)

    # 初始化评价类
    cfgs = {
        "class_to_index": {"T": 0, "L": 1},
        "confidence_threshold": 0.5,
        "distance_threshold": 0.1,
        "angle_threshold": 5,
        "consider_class": True,
    }
    evaluator = DirectionalCornerDetectionMetric(cfgs)

    # 更新指标
    evaluator.update(pred, targets)

    # 计算指标
    metrics = evaluator.compute()

    print(f"准确率: {metrics['accuracy']}")
    print(f"精确率: {metrics['precision']}")
    print(f"召回率: {metrics['recall']}")
    print(f"F1 分数: {metrics['f1_score']}")
    print(f"置信度均值: {metrics['confidence_mean']}")
    # print(f"置信度标准差: {metrics['confidence_std']}")
    print(f"类别分布: {metrics['class_distribution']}")
    print(f"角度误差均值: {metrics['angle_error_mean']}")
    print(f"角度误差最大值: {metrics['angle_error_max']}")
    print(f"中心坐标误差均值: {metrics['center_error_mean']}")
    print(f"中心坐标误差最大值: {metrics['center_error_max']}")
    # print(f"mAP: {metrics['map']}")
