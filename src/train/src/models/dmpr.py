"""Defines the detector network structure."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import  models

def define_squeeze_unit(basic_channel_size):
    """Define a 1x1 squeeze convolution with norm and activation."""
    conv = nn.Conv2d(
        2 * basic_channel_size,
        basic_channel_size,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    norm = nn.BatchNorm2d(basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv = nn.Conv2d(
        basic_channel_size,
        2 * basic_channel_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_halve_unit(basic_channel_size):
    """Define a 4x4 stride 2 expand convolution with norm and activation."""
    conv = nn.Conv2d(
        basic_channel_size,
        2 * basic_channel_size,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
    )
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_depthwise_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv1 = nn.Conv2d(
        basic_channel_size,
        2 * basic_channel_size,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    norm1 = nn.BatchNorm2d(2 * basic_channel_size)
    relu1 = nn.LeakyReLU(0.1)
    conv2 = nn.Conv2d(
        2 * basic_channel_size,
        2 * basic_channel_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=2 * basic_channel_size,
    )
    norm2 = nn.BatchNorm2d(2 * basic_channel_size)
    relu2 = nn.LeakyReLU(0.1)
    layers = [conv1, norm1, relu1, conv2, norm2, relu2]
    return layers


def define_detector_block(basic_channel_size):
    """Define a unit composite of a squeeze and expand unit."""
    layers = []
    layers += define_squeeze_unit(basic_channel_size)
    layers += define_expand_unit(basic_channel_size)
    return layers


class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""

    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [
            nn.Conv2d(
                input_channel_size,
                depth_factor,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        ]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        # 1
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 2
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        # 3
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 4
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        layers += define_detector_block(depth_factor)
        # 5
        depth_factor *= 2
        layers += define_halve_unit(depth_factor)
        layers += define_detector_block(depth_factor)
        self.model = nn.Sequential(*layers)

    def forward(self, *x):
        return self.model(x[0])


class DirectionalPointDetector(nn.modules.Module):
    """Detector for point with direction."""

    def __init__(self, input_channel_size, depth_factor, output_channel_size):
        super(DirectionalPointDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(
            input_channel_size, depth_factor
        )
        layers = []
        layers += define_detector_block(16 * depth_factor)
        layers += define_detector_block(16 * depth_factor)
        layers += [
            nn.Conv2d(
                32 * depth_factor,
                output_channel_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        self.predict = nn.Sequential(*layers)

    def forward(self, *x):
        prediction = self.predict(self.extract_feature(x[0]))
        # 4 represents that there are 4 value: confidence, shape, offset_x,
        # offset_y, whose range is between [0, 1].
        total_dim = prediction.shape[1]  # 总维度
        remaining_dim = total_dim - 5  # 剩余维度
        point_pred, angle_pred,class_channels = torch.split(prediction, [3,2,remaining_dim], dim=1)
        point_pred = torch.sigmoid(point_pred)
        angle_pred = torch.tanh(angle_pred)
        class_channels = F.softmax(class_channels, dim=1)

        return torch.cat((point_pred, angle_pred,class_channels), dim=1)
