import torch
import torch.nn as nn
import torch.nn.functional as F

    
# class DirectionalCornerDetectionLoss(nn.Module):
#     def __init__(self, num_classes):
#         super(DirectionalCornerDetectionLoss, self).__init__()
#         self.num_classes = num_classes
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, pred, targets):
#         """
#         pred: [batch_size, 5 + num_classes, grid_size, grid_size]
#         targets: [batch_size, 5 + num_classes, grid_size, grid_size]
#         """
#         # 提取预测值
#         confidence_pred = pred[:, 0, :, :]
#         x_pred = pred[:, 1, :, :]
#         y_pred = pred[:, 2, :, :]
#         cos_pred = pred[:, 3, :, :]
#         sin_pred = pred[:, 4, :, :]
#         class_probs_pred = pred[:, 5:, :, :]

#         # 提取目标值
#         confidence_target = targets[:, 0, :, :]
#         x_target = targets[:, 1, :, :]
#         y_target = targets[:, 2, :, :]
#         cos_target = targets[:, 3, :, :]
#         sin_target = targets[:, 4, :, :]
#         class_probs_target = targets[:, 5:, :, :]

#         # 计算中心坐标损失
#         x_loss = self.mse(x_pred, x_target)
#         y_loss = self.mse(y_pred, y_target)

#         # 计算cos和sin损失
#         cos_loss = self.mse(cos_pred, cos_target)
#         sin_loss = self.mse(sin_pred, sin_target)

#         # 计算置信度损失
#         confidence_loss = self.bce(confidence_pred, confidence_target)

#         # 计算类别概率损失
#         # 将 class_probs_pred 调整为 (N, C) 形状
#         class_probs_pred = class_probs_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
#         # 将 class_probs_target 转换为类别索引
#         class_probs_target = torch.argmax(class_probs_target.permute(0, 2, 3, 1), dim=-1).reshape(-1)
#         class_loss = self.ce(class_probs_pred, class_probs_target)

#         # 总损失
#         total_loss = x_loss + y_loss + cos_loss + sin_loss + confidence_loss + class_loss

#         return total_loss


class DirectionalCornerDetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super(DirectionalCornerDetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, targets):
        """
        pred: [batch_size, 5 + num_classes, grid_size, grid_size]
        targets: [batch_size, 5 + num_classes, grid_size, grid_size]
        """
        batch_size, _, grid_size, _ = pred.shape

        # 提取预测值
        confidence_pred = pred[:, 0, :, :]
        x_pred = pred[:, 1, :, :]
        y_pred = pred[:, 2, :, :]
        cos_pred = pred[:, 3, :, :]
        sin_pred = pred[:, 4, :, :]
        class_probs_pred = pred[:, 5:, :, :]

        # 提取目标值
        confidence_target = targets[:, 0, :, :]
        x_target = targets[:, 1, :, :]
        y_target = targets[:, 2, :, :]
        cos_target = targets[:, 3, :, :]
        sin_target = targets[:, 4, :, :]
        class_probs_target = targets[:, 5:, :, :]

        # 找到置信度为1的网格位置
        valid_mask = (confidence_target >= 0.5).float()  # 置信度阈值可以根据需要调整

        # 筛选预测值和目标值
        x_pred_valid = x_pred * valid_mask
        y_pred_valid = y_pred * valid_mask
        cos_pred_valid = cos_pred * valid_mask
        sin_pred_valid = sin_pred * valid_mask
        class_probs_pred_valid = class_probs_pred * valid_mask.unsqueeze(1)

        x_target_valid = x_target * valid_mask
        y_target_valid = y_target * valid_mask
        cos_target_valid = cos_target * valid_mask
        sin_target_valid = sin_target * valid_mask
        class_probs_target_valid = class_probs_target * valid_mask.unsqueeze(1)

        # 计算中心坐标损失
        x_loss = self.mse(x_pred_valid, x_target_valid)
        y_loss = self.mse(y_pred_valid, y_target_valid)

        # 计算cos和sin损失
        cos_loss = self.mse(cos_pred_valid, cos_target_valid)
        sin_loss = self.mse(sin_pred_valid, sin_target_valid)

        # 计算置信度损失
        # confidence_loss = self.bce(confidence_pred, confidence_target)
        confidence_loss = self.mse(confidence_pred, confidence_target)

        # 计算类别概率损失
        # 将 class_probs_pred 调整为 (N, C) 形状
        class_probs_pred_valid = class_probs_pred_valid.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        # 将 class_probs_target 转换为类别索引
        class_probs_target_valid = torch.argmax(class_probs_target_valid.permute(0, 2, 3, 1), dim=-1).reshape(-1)
        class_loss = self.ce(class_probs_pred_valid, class_probs_target_valid)

        # 总损失
        total_loss = x_loss + y_loss + cos_loss + sin_loss + confidence_loss + class_loss
        # total_loss = confidence_loss

        return total_loss
