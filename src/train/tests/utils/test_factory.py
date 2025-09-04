import torch.nn as nn

import pytest

from lovely_deep_learning.utils.factory import dynamic_class_instantiate_from_string


def test_dynamic_class_instantiate_from_string():
    test_cases = [
        (
            "torch.nn.Conv2d",
            {"in_channels": 3, "out_channels": 64, "kernel_size": 3},
            nn.Conv2d,
        ),
        ("torch.nn.BatchNorm2d", {"num_features": 64}, nn.BatchNorm2d),
        ("torch.nn.ReLU", {"inplace": True}, nn.ReLU),
        ("torch.nn.MaxPool2d", {"kernel_size": 2, "stride": 2}, nn.MaxPool2d),
    ]

    for class_path, kwargs, expected_class in test_cases:
        instance = dynamic_class_instantiate_from_string(class_path, **kwargs)
        assert isinstance(instance, expected_class)
