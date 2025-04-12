import torch
import torch.nn as nn
import torchvision.models as models
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_feature_extractor(backbone, backbone_name):
    """
    构建特征提取器
    :param backbone: 骨干网络模型
    :param backbone_name: 骨干网络名称
    :return: 特征提取器
    """
    if 'resnet' in backbone_name or 'densenet' in backbone_name:
        return nn.Sequential(*list(backbone.children())[:-2])
    elif 'vgg' in backbone_name:
        return nn.Sequential(*list(backbone.features.children()))
    elif 'mobilenet' in backbone_name or 'efficientnet' in backbone_name or 'shufflenet' in backbone_name or 'squeezenet' in backbone_name:
        return backbone.features
    elif 'inception' in backbone_name:
        return nn.Sequential(
            backbone.Conv2d_1a_3x3,
            backbone.Conv2d_2a_3x3,
            backbone.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            backbone.Conv2d_3b_1x1,
            backbone.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            backbone.Mixed_5b,
            backbone.Mixed_5c,
            backbone.Mixed_5d,
            backbone.Mixed_6a,
            backbone.Mixed_6b,
            backbone.Mixed_6c,
            backbone.Mixed_6d,
            backbone.Mixed_6e,
            backbone.Mixed_7a,
            backbone.Mixed_7b,
            backbone.Mixed_7c
        )
    elif 'googlenet' in backbone_name:
        return nn.Sequential(
            backbone.conv1,
            backbone.maxpool1,
            backbone.conv2,
            backbone.conv3,
            backbone.maxpool2,
            backbone.inception3a,
            backbone.inception3b,
            backbone.maxpool3,
            backbone.inception4a,
            backbone.inception4b,
            backbone.inception4c,
            backbone.inception4d,
            backbone.inception4e,
            backbone.maxpool4,
            backbone.inception5a,
            backbone.inception5b
        )
    elif 'mnasnet' in backbone_name:
        return backbone.layers
    elif 'regnet' in backbone_name:
        return nn.Sequential(
            backbone.stem,
            backbone.trunk_output
        )
    elif 'vit' in backbone_name:
        return nn.Sequential(
            backbone.conv_proj,
            backbone.encoder
        )
    elif 'swin' in backbone_name:
        return nn.Sequential(
            backbone.patch_embed,
            backbone.pos_drop,
            backbone.layers,
            backbone.norm
        )
    elif 'maxvit' in backbone_name:
        return nn.Sequential(
            backbone.stem,
            backbone.stage1,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4
        )
    else:
        logging.error(f"Unsupported backbone for feature extraction: {backbone_name}")
        raise ValueError(f"Unsupported backbone for feature extraction: {backbone_name}")

class Cnn_Backbone(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, num_classes=10):
        super(Cnn_Backbone, self).__init__()
        self.backbone_name = backbone_name
        try:
            # 直接使用 getattr 从 torchvision.models 中获取模型构建函数
            model_builder = getattr(models, backbone_name)
            self.backbone = model_builder(pretrained=pretrained)
        except AttributeError:
            logging.error(f"Unsupported backbone: {backbone_name}")
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 构建特征提取器
        self.feature_extractor = build_feature_extractor(self.backbone, backbone_name)

    def forward(self, x):
        # 通过特征提取层进行特征提取
        features = self.feature_extractor(x)
        return features


# 示例使用
if __name__ == "__main__":
    # 创建一个使用 ResNet18 作为骨干网络的完整模型实例
    model = Cnn_Backbone(backbone_name='resnet18', pretrained=True, num_classes=10)
    # 生成一个随机输入张量，模拟图像数据
    input_tensor = torch.randn(1, 3, 224, 224)
    # 前向传播，获取预测结果
    output = model(input_tensor)
    print("Output shape:", output.shape)
    