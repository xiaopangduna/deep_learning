from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _make_divisible(v: int | float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class Conv2dNormActivation(nn.Sequential):
    """Conv-BN-(Act) block with torchvision-compatible key layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,
        activation_layer: str | None = "relu",
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers: list[nn.Module] = [
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
            nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
        ]
        if activation_layer == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation_layer == "hswish":
            layers.append(nn.Hardswish(inplace=True))
        elif activation_layer is None:
            pass
        else:
            raise ValueError(f"Unsupported activation_layer: {activation_layer}")
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """SE block matching torchvision key names fc1/fc2."""

    def __init__(self, *, input_channels: int, squeeze_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale(x) * x


class InvertedResidual(nn.Module):
    """MobileNetV3 inverted residual block with torchvision-compatible structure."""

    def __init__(
        self,
        *,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        use_hs: bool,
        stride: int,
        dilation: int,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels
        activation = "hswish" if use_hs else "relu"

        layers: list[nn.Module] = []

        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    activation_layer=activation,
                )
            )

        depthwise_stride = 1 if dilation > 1 else stride
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel,
                stride=depthwise_stride,
                dilation=dilation,
                groups=expanded_channels,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
                activation_layer=activation,
            )
        )

        if use_se:
            squeeze_channels = _make_divisible(expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(input_channels=expanded_channels, squeeze_channels=squeeze_channels))

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

