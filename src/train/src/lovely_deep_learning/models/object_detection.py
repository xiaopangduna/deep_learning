import torch
from torch import nn
import torch.nn.functional as F
from src.models.backbone import Cnn_Backbone

class DirectionalCornerDetectionModel(nn.Module):
    def __init__(self, backbone_name, pretrained, input_size,output_size):
        super(DirectionalCornerDetectionModel, self).__init__()
        self.extract_feature = Cnn_Backbone(backbone_name=backbone_name,pretrained=pretrained)
        self.direction_corner = DirectionalCornerOutputLayer(output_size[1]-5)
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            features = self.extract_feature(dummy_input)
            in_channels = features.size(1)
            # 设置输出层的输入通道数
            self.direction_corner.set_in_channels(in_channels)

    def forward(self, *x):
        extract_feature = self.extract_feature(x[0])
        preds = self.direction_corner(extract_feature)
        total_dim = preds.shape[1]  # 总维度
        remaining_dim = total_dim - 5  # 剩余维度
        point_pred, angle_pred,class_channels = torch.split(preds, [3,2,remaining_dim], dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        class_channels = F.softmax(class_channels, dim=1)
        return torch.cat((point_pred, angle_pred,class_channels), dim=1)
class DirectionalCornerOutputLayer(nn.Module):
    def __init__(self, num_classes):
        super(DirectionalCornerOutputLayer, self).__init__()
        # 这里简单使用一个 1x1 卷积层作为输出层
        self.num_classes = num_classes
        self.conv = None
    def set_in_channels(self, in_channels):
        # 在获取到输入通道数后，初始化卷积层
        # 5 = 置信度，x，y，cos，sin
        self.conv = nn.Conv2d(in_channels, 5 + self.num_classes, kernel_size=1)
       
    def forward(self, x):
        if self.conv is None:
            raise ValueError("Input channels are not set. Call set_in_channels first.")
        output = self.conv(x)
        return output
