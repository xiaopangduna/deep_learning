from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class Conv2dNormActivation(nn.Sequential):
    """Conv-BN-(Act) block with torchvision MobileNetV2-compatible key layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        bn_eps: float = 1e-5,
        activation_layer: str | None = "relu6",
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
            nn.BatchNorm2d(out_channels, eps=bn_eps),
        ]
        if activation_layer == "relu6":
            layers.append(nn.ReLU6(inplace=True))
        elif activation_layer == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation_layer is None:
            pass
        else:
            raise ValueError(f"Unsupported activation_layer: {activation_layer}")
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block with torchvision-compatible structure."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        expand_ratio: int,
        stride: int,
        bn_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(input_channels * expand_ratio))
        self.use_res_connect = stride == 1 and input_channels == output_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    bn_eps=bn_eps,
                    activation_layer="relu6",
                )
            )
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    bn_eps=bn_eps,
                    activation_layer="relu6",
                ),
                nn.Conv2d(hidden_dim, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels, eps=bn_eps),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = output_channels
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)
