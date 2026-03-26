import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) // 2 * dilation


def _conv_bn(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    eps: float = 1e-3,
) -> nn.Sequential:
    padding = _same_padding(kernel_size, dilation=dilation)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=eps),
    )


def _conv_bn_act(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    eps: float = 1e-3,
    inplace: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        *_conv_bn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation,
            eps=eps,
        ),
        nn.SiLU(inplace=inplace),
    )


class SqueezeExcitationSiLU(nn.Module):
    """
    TorchVision SqueezeExcitation equivalent with fixed activations:
    - activation: SiLU(inplace=True)
    - scale_activation: Sigmoid()
    """

    def __init__(self, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(x) * x


class MBConv(nn.Module):
    """
    YAML-friendly EfficientNetV2 MBConv block mirroring TorchVision's module/key structure.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        out_channels: int,
        expand_ratio: float,
        kernel: int,
        stride: int,
        stochastic_depth_prob: float,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: list[nn.Module] = []

        expanded_channels = int(math.ceil(input_channels * expand_ratio))
        # TorchVision uses _make_divisible; for V2-S settings channels are already aligned.
        # Keep expanded_channels deterministic for exact key/shape matching.

        if expanded_channels != input_channels:
            layers.append(
                _conv_bn_act(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                    eps=bn_eps,
                )
            )

        layers.append(
            _conv_bn_act(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=stride,
                groups=expanded_channels,
                eps=bn_eps,
            )
        )

        squeeze_channels = max(1, input_channels // 4)
        layers.append(SqueezeExcitationSiLU(expanded_channels, squeeze_channels))

        layers.append(
            _conv_bn(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                eps=bn_eps,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    """
    YAML-friendly EfficientNetV2 FusedMBConv block mirroring TorchVision's module/key structure.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        out_channels: int,
        expand_ratio: float,
        kernel: int,
        stride: int,
        stochastic_depth_prob: float,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: list[nn.Module] = []

        expanded_channels = int(math.ceil(input_channels * expand_ratio))

        if expanded_channels != input_channels:
            layers.append(
                _conv_bn_act(
                    input_channels,
                    expanded_channels,
                    kernel_size=kernel,
                    stride=stride,
                    eps=bn_eps,
                )
            )
            layers.append(
                _conv_bn(
                    expanded_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    eps=bn_eps,
                )
            )
        else:
            layers.append(
                _conv_bn_act(
                    input_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    eps=bn_eps,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

