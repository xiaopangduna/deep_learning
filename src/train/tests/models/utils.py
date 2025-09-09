import yaml


def load_yaml_config(file_path):
    try:
        # 用with语句自动管理文件关闭
        with open(file_path, "r", encoding="utf-8") as f:
            # safe_load() 安全解析YAML内容
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except yaml.YAMLError as e:
        print(f"YAML解析错误：{e}")
        return None


resnet18_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 224, 224]}],
        "outputs": [{"name": "classification", "from": ["fc"]}],
        "layers": [
            # Stem
            {
                "name": "conv1",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 64,
                    "kernel_size": 7,
                    "stride": 2,
                    "padding": 3,
                    "bias": False,
                },
                "from": ["input"],
            },
            {
                "name": "bn1",
                "module": "torch.nn.BatchNorm2d",
                "args": {"num_features": 64},
                "from": ["conv1"],
            },
            {
                "name": "relu",
                "module": "torch.nn.ReLU",
                "args": {"inplace": True},
                "from": ["bn1"],
            },
            {
                "name": "maxpool",
                "module": "torch.nn.MaxPool2d",
                "args": {"kernel_size": 3, "stride": 2, "padding": 1},
                "from": ["relu"],
            },
            # Layer1
            {
                "name": "layer1",
                "module": "torch.nn.Sequential",
                "from": ["maxpool"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
                    },
                ],
            },
            # Layer2
            {
                "name": "layer2",
                "module": "torch.nn.Sequential",
                "from": ["layer1"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 64, "out_channels": 128, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 128, "out_channels": 128, "stride": 1},
                    },
                ],
            },
            # Layer3
            {
                "name": "layer3",
                "module": "torch.nn.Sequential",
                "from": ["layer2"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 128, "out_channels": 256, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 256, "out_channels": 256, "stride": 1},
                    },
                ],
            },
            # Layer4
            {
                "name": "layer4",
                "module": "torch.nn.Sequential",
                "from": ["layer3"],
                "children": [
                    {
                        "name": "0",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 256, "out_channels": 512, "stride": 2},
                    },
                    {
                        "name": "1",
                        "module": "lovely_deep_learning.nn.conv.BasicBlock",
                        "args": {"in_channels": 512, "out_channels": 512, "stride": 1},
                    },
                ],
            },
            # Head
            {
                "name": "avgpool",
                "module": "torch.nn.AdaptiveAvgPool2d",
                "args": {"output_size": [1, 1]},
                "from": ["layer4"],
            },
            {
                "name": "flatten",
                "module": "torch.nn.Flatten",
                "args": {},
                "from": ["avgpool"],
            },
            {
                "name": "fc",
                "module": "torch.nn.Linear",
                "args": {"in_features": 512, "out_features": 1000},
                "from": ["flatten"],
            },
        ],
    },
    "weight": {
        "path": "pretrained_models/resnet18-f37072fd.pth",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "map_location": "cpu",
        "strict": False,
    },
}


demo_config_two_inputs_one_outputs = {
    "structure": {
        "inputs": [
            {"name": "img1", "shape": (3, 32, 32)},
            {"name": "img2", "shape": (3, 32, 32)},
        ],
        "outputs": [{"name": "class", "shape": (10,), "from": ["fc"]}],
        "layers": [
            {
                "name": "conv1",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["img1"],
            },
            {
                "name": "conv2",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 3,
                    "out_channels": 16,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["img2"],
            },
            {
                "name": "concat",
                "module": "lovely_deep_learning.nn.conv.Concat",
                "args": {"dim": 1},
                "from": ["conv1", "conv2"],
            },
            {
                "name": "conv3",
                "module": "torch.nn.Conv2d",
                "args": {
                    "in_channels": 32,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1,
                },
                "from": ["concat"],
            },
            {
                "name": "flatten",
                "module": "torch.nn.Flatten",
                "args": {},
                "from": ["conv3"],
            },
            {
                "name": "fc",
                "module": "torch.nn.LazyLinear",
                "args": {"out_features": 10},
                "from": ["flatten"],
            },
        ],
    },
}
