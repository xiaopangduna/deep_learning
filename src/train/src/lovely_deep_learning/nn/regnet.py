from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class Conv2dNormActivation(nn.Sequential):
    """
    Minimal Conv-BN-(Act) with key layout matching torchvision.ops.misc.Conv2dNormActivation:
    - conv at index 0
    - bn at index 1
    - act at index 2 (no params)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = False,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SimpleStemIN(Conv2dNormActivation):
    """ImageNet stem: 3x3, BN, ReLU, stride=2."""

    def __init__(
        self,
        *,
        width_in: int = 3,
        width_out: int = 32,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            activation=True,
        )


class SqueezeExcitation(nn.Module):
    """
    SE block with key layout matching torchvision.ops.misc.SqueezeExcitation:
    avgpool, fc1, fc2, activation, scale_activation
    """

    def __init__(self, *, input_channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(x) * x


class BottleneckTransform(nn.Sequential):
    """1x1, 3x3 (grouped) + optional SE, 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        *,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            activation=True,
        )
        layers["b"] = Conv2dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=g,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            activation=True,
        )

        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(input_channels=w_b, squeeze_channels=width_se_out)

        layers["c"] = Conv2dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            activation=False,
        )

        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual: x + F(x), optional projection."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        *,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.proj: Optional[nn.Module] = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=1,
                bias=False,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
                activation=False,
            )

        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            se_ratio=se_ratio,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """Stage with blocks named block{stage_index}-{i} to match torchvision keys."""

    def __init__(
        self,
        *,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        stage_index: int = 1,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        for i in range(depth):
            block = ResBottleneckBlock(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                group_width=group_width,
                bottleneck_multiplier=bottleneck_multiplier,
                se_ratio=se_ratio,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
            self.add_module(f"block{stage_index}-{i}", block)

