import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
from sklearn.metrics import precision_score, recall_score

# class DirectionalCornerDetectionMetric:
#     def __init__(self, num_classes, distance_threshold=0.1, angle_threshold=5, confidence_threshold=0.5, consider_class=True):
#         self.num_classes = num_classes
#         self.distance_threshold = distance_threshold
#         self.angle_threshold = angle_threshold
#         self.confidence_threshold = confidence_threshold
#         self.consider_class = consider_class

#     def evaluate(self, pred, targets):
#         """
#         pred: [batch_size, grid_size, grid_size, 5 + num_classes]
#         targets: [batch_size, grid_size, grid_size, 5 + num_classes]
#         """
#         # 提取预测值
#         confidence_pred = pred[..., 0]
#         x_pred = pred[..., 1]
#         y_pred = pred[..., 2]
#         cos_pred = pred[..., 3]
#         sin_pred = pred[..., 4]
#         class_probs_pred = pred[..., 5:]

#         # 提取目标值
#         confidence_target = targets[..., 0]
#         x_target = targets[..., 1]
#         y_target = targets[..., 2]
#         cos_target = targets[..., 3]
#         sin_target = targets[..., 4]
#         class_probs_target = targets[..., 5:]

#         # 计算匹配率
#         match_rate = self.calculate_match_rate(confidence_pred, x_pred, y_pred, cos_pred, sin_pred, class_probs_pred, confidence_target, x_target, y_target, cos_target, sin_target, class_probs_target)

#         return match_rate

#     def calculate_match_rate(self, confidence_pred, x_pred, y_pred, cos_pred, sin_pred, class_probs_pred, confidence_target, x_target, y_target, cos_target, sin_target, class_probs_target):
#         # 计算置信度匹配
#         confidence_match = confidence_pred > self.confidence_threshold

#         # 计算中心坐标的距离
#         distance = torch.sqrt((x_pred - x_target) ** 2 + (y_pred - y_target) ** 2)
#         distance_match = distance < self.distance_threshold

#         # 计算角度误差
#         angle_pred = torch.atan2(sin_pred, cos_pred)
#         angle_target = torch.atan2(sin_target, cos_target)
#         angle_error = torch.abs(angle_pred - angle_target)
#         # 将角度误差限制在 [-180, 180] 范围内
#         angle_error = torch.min(angle_error, 360 - angle_error)
#         # 转换为度数
#         angle_error = angle_error * 180 / np.pi
#         # 判断角度误差是否小于阈值
#         angle_match = angle_error < self.angle_threshold

#         # 计算类别匹配
#         _, predicted = torch.max(class_probs_pred, dim=-1)
#         _, target = torch.max(class_probs_target, dim=-1)
#         class_match = predicted == target

#         # 判断匹配
#         if self.consider_class:
#             match = confidence_match & distance_match & angle_match & class_match
#         else:
#             match = confidence_match & distance_match & angle_match
#         match_rate = match.float().mean()

#         return match_rate
    

class DirectionalCornerDetectionMetric(torchmetrics.Metric):
    def __init__(self, cfgs: dict):
        super().__init__()
        self.class_to_index = cfgs["class_to_index"]
        self.classes = [0] * len(self.class_to_index)
        for key in self.class_to_index.keys():
            self.classes[self.class_to_index[key]] = key
        self.confidence_threshold = cfgs.get("confidence_threshold", 0.5)
        self.distance_threshold = cfgs.get("distance_threshold", 0.1)
        self.angle_threshold = cfgs.get("angle_threshold", 5)
        self.consider_class = cfgs.get("consider_class", False)
        self.classes = cfgs.get("classes", ["L","T"])
        # 统计预测正确，错误，漏检的数量
        self.counts = torch.zeros((len(self.classes), 3), dtype=torch.long)
        # 初始化指标
        # self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        # self.precision = torchmetrics.Precision(task="multiclass", num_classes=2,average="micro")
        # self.recall = torchmetrics.Recall(task="multiclass", num_classes=2)
        # self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
        self.confidence_mean = torchmetrics.MeanMetric()
        self.class_distribution = {}
        self.angle_error_mean = torchmetrics.MeanMetric()
        self.angle_error_max = torchmetrics.MaxMetric()
        self.center_error_mean = torchmetrics.MeanMetric()
        self.center_error_max = torchmetrics.MaxMetric()


    def update(self, pred: torch.Tensor, targets: torch.Tensor):
        # 确保张量形状为 [batch_size, 5 + num_classes, grid_size, grid_size]

        # 提取预测值
        confidence_pred = pred[:, 0, :, :]
        pred_mask = confidence_pred > self.confidence_threshold
        x_pred = pred[:, 1, :, :]
        y_pred = pred[:, 2, :, :]
        cos_pred = pred[:, 3, :, :]
        sin_pred = pred[:, 4, :, :]
        class_probs_pred = pred[:, 5:, :, :]

        # 提取目标值
        confidence_target = targets[:, 0, :, :]
        target_mask = confidence_target > self.confidence_threshold
        x_target = targets[:, 1, :, :]
        y_target = targets[:, 2, :, :]
        cos_target = targets[:, 3, :, :]
        sin_target = targets[:, 4, :, :]
        class_probs_target = targets[:, 5:, :, :]

        # 计算匹配
        match = self.calculate_match(confidence_pred, x_pred, y_pred, cos_pred, sin_pred, class_probs_pred, confidence_target, x_target, y_target, cos_target, sin_target, class_probs_target)
        # 计算TP，FP，TN，FN
        # 更新指标
        # self.precision.update(match, torch.ones_like(match))
        # self.recall.update(match, torch.ones_like(match))
        # self.f1_score.update(match, torch.ones_like(match))
        self.confidence_mean.update(confidence_pred)
        self.angle_error_mean.update(self.calculate_angle_error(match))
        self.angle_error_max.update(self.calculate_angle_error(match))
        self.center_error_mean.update(self.calculate_center_error(match))
        self.center_error_max.update(self.calculate_center_error(match))
        # self.map.update(match, torch.ones_like(match))

        # 更新类别分布
        # counts[num_classes,3],3 = [TP，FP，FN，TN]
        counts= self.update_class_distribution(class_probs_pred, class_probs_target,match,pred_mask,target_mask)
        self.counts += counts

    def compute(self):

        column_sum = torch.sum(self.counts, dim=0)
        # 提取 TP、FP、FN
        TP = column_sum[0]
        FP = column_sum[1]
        FN = column_sum[2]
        # 已知 TN = 0
        TN = torch.tensor(0)
        # 计算分类指标
        # 准确率
        if (TP + TN + FP + FN) == 0:
            accuracy = torch.tensor(0.0)
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        # 精确率
        if (TP + FP) == 0:
            precision = torch.tensor(0.0)
        else:
            precision = TP / (TP + FP)
        # 召回率
        if (TP + FN) == 0:
            recall = torch.tensor(0.0)
        else:
            recall = TP / (TP + FN)

        # F1 分数
        if (precision + recall) == 0:
            f1_score = torch.tensor(0.0)
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        return {
            "accuracy":accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "TP":TP,
            "FP":FP,
            "FN":FN,
            "confidence_mean": self.confidence_mean.compute(),
            # "confidence_std": self.confidence_std.compute(),
            # "class_distribution": self.class_distribution,
            "angle_error_mean": self.angle_error_mean.compute(),
            "angle_error_max": self.angle_error_max.compute(),
            "center_error_mean": self.center_error_mean.compute(),
            "center_error_max": self.center_error_max.compute(),
            # "map": self.map.compute()
        }

    def calculate_match(self, confidence_pred, x_pred, y_pred, cos_pred, sin_pred, class_probs_pred, confidence_target, x_target, y_target, cos_target, sin_target, class_probs_target):
        # 计算置信度匹配
        confidence_match = confidence_pred > self.confidence_threshold

        # 计算中心坐标的距离
        distance = torch.sqrt((x_pred - x_target) ** 2 + (y_pred - y_target) ** 2)
        distance_match = distance < self.distance_threshold

        # 计算角度误差
        angle_pred = torch.atan2(sin_pred, cos_pred)
        angle_target = torch.atan2(sin_target, cos_target)
        angle_error = torch.abs(angle_pred - angle_target)
        angle_error = torch.min(angle_error, 360 - angle_error)
        angle_error = angle_error * 180 / np.pi
        angle_match = angle_error < self.angle_threshold

        # 计算类别匹配
        _, predicted = torch.max(class_probs_pred, dim=1)
        _, target = torch.max(class_probs_target, dim=1)
        class_match = predicted == target

        # 判断匹配
        if self.consider_class:
            match = confidence_match & distance_match & angle_match & class_match
        else:
            match = confidence_match & distance_match & angle_match

        return match.float()

    def calculate_angle_error(self, predicted_value):
        # 计算角度误差均值
        angle_error = predicted_value
        return angle_error

    def calculate_center_error(self, predicted_value):
        # 计算中心坐标误差均值
        center_error = predicted_value
        return center_error

    def reset(self):
        self.counts.fill_(0)
        # 重置指标
        # self.accuracy.reset()
        # self.precision.reset()
        # self.recall.reset()
        # self.f1_score.reset()
        # self.confidence_mean.reset()
        # # self.confidence_std.reset()
        # self.class_distribution = {}
        # self.angle_error_mean.reset()
        # self.angle_error_max.reset()
        # self.center_error_mean.reset()
        # self.center_error_max.reset()
        # self.map.reset()
        pass

    def update_class_distribution(self,pred_classes, target_classes,match,pred_mask, target_mask):
        # 计算预测类别和真实类别
        # _, predicted = torch.max(pred_classes, dim=1)
        # _, target = torch.max(target_classes, dim=1)
        # 去掉置信度，只保留类别信息
        # pred_classes = pred[:, 1:, :, :]
        # target_classes = target[:, 1:, :, :]
        match = match.bool()
        num_classes = pred_classes.shape[1]
        predicted_classes = torch.argmax(pred_classes, dim=1)
        true_classes = torch.argmax(target_classes, dim=1)

        # 初始化每个类别的统计结果
        metrics = torch.zeros((num_classes, 3), dtype=torch.long)

        # 遍历每个类别
        for class_idx in range(num_classes):
            # 应用预测掩码
            class_pred_mask = (predicted_classes == class_idx) & pred_mask
            # 应用真实掩码
            class_target_mask = (true_classes == class_idx) & target_mask

            # 统计预测正确的数量
            correct = (class_pred_mask & class_target_mask & match).sum()

            # 统计预测错误的数量
            wrong = (class_pred_mask & ~class_target_mask).sum()

            # 统计漏检的数量
            missed = (~class_pred_mask & class_target_mask).sum()

            # 将统计结果存储到张量中
            metrics[class_idx, 0] = correct
            metrics[class_idx, 1] = wrong
            metrics[class_idx, 2] = missed

        return metrics


# def calculate_class_metrics(metrics):
#     num_classes = metrics.shape[0]
#     precision = torch.zeros(num_classes)
#     recall = torch.zeros(num_classes)
#     f1_score = torch.zeros(num_classes)

#     for i in range(num_classes):
#         tp = metrics[i, 0].item()  # 预测正确数量
#         fp = metrics[i, 1].item()  # 预测错误数量
#         fn = metrics[i, 2].item()  # 漏检数量

#         # 计算精确率
#         if tp + fp > 0:
#             precision[i] = tp / (tp + fp)
#         else:
#             precision[i] = 0

#         # 计算召回率
#         if tp + fn > 0:
#             recall[i] = tp / (tp + fn)
#         else:
#             recall[i] = 0

#         # 计算 F1 分数
#         if precision[i] + recall[i] > 0:
#             f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
#         else:
#             f1_score[i] = 0

#     return precision, recall, f1_score

# def calculate_total_metrics(metrics):
#     # 计算宏平均
#     precision, recall, f1_score = calculate_class_metrics(metrics)
#     macro_precision = precision.mean().item()
#     macro_recall = recall.mean().item()
#     macro_f1 = f1_score.mean().item()

#     # 计算微平均
#     total_tp = metrics[:, 0].sum().item()
#     total_fp = metrics[:, 1].sum().item()
#     total_fn = metrics[:, 2].sum().item()

#     if total_tp + total_fp > 0:
#         micro_precision = total_tp / (total_tp + total_fp)
#     else:
#         micro_precision = 0

#     if total_tp + total_fn > 0:
#         micro_recall = total_tp / (total_tp + total_fn)
#     else:
#         micro_recall = 0

#     if micro_precision + micro_recall > 0:
#         micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
#     else:
#         micro_f1 = 0

#     return {
#         'Macro Precision': macro_precision,
#         'Macro Recall': macro_recall,
#         'Macro F1': macro_f1,
#         'Micro Precision': micro_precision,
#         'Micro Recall': micro_recall,
#         'Micro F1': micro_f1
#     }

# # 示例 metrics 张量
# metrics = torch.tensor([
#     [10, 2, 3],  # 类别 0 的统计结果
#     [15, 4, 5],  # 类别 1 的统计结果
#     [20, 6, 7]   # 类别 2 的统计结果
# ], dtype=torch.float32)

# # 计算总的分类指标
# total_metrics = calculate_total_metrics(metrics)

# # 输出结果
# for metric, value in total_metrics.items():
#     print(f"{metric}: {value:.4f}")