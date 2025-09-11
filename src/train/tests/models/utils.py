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


yolov8_n_config = {
    "structure": {
        "inputs": [{"name": "input", "shape": [3, 640, 640]}],
        "outputs": [{"name": "detect", "from": ["13"]}],
        "layers": [
            # Stem
            {
                "name": "0",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 3,
                    "c2": 16,
                    "k": 3,
                    "s": 2,
                },
                "from": ["input"],
            },
            {
                "name": "1",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 16,
                    "c2": 32,
                    "k": 3,
                    "s": 2,
                },
                "from": ["0"],
            },
            {
                "name": "2",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 32,
                    "c2": 32,
                    "n": 1,
                    "shortcut": True,
                },
                "from": ["1"],
            },
            {
                "name": "3",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 32,
                    "c2": 64,
                    "k": 3,
                    "s": 2,
                },
                "from": ["2"],
            },
            {
                "name": "4",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 64,
                    "c2": 64,
                    "n": 2,
                    "shortcut": True,
                },
                "from": ["3"],
            },
            {
                "name": "5",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 64,
                    "c2": 128,
                    "k": 3,
                    "s": 2,
                },
                "from": ["4"],
            },
            {
                "name": "6",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 128,
                    "c2": 128,
                    "n": 2,
                    "shortcut": True,
                },
                "from": ["5"],
            },
            {
                "name": "7",
                "module": "lovely_deep_learning.nn.conv.Conv",
                "args": {
                    "c1": 128,
                    "c2": 256,
                    "k": 3,
                    "s": 2,
                },
                "from": ["6"],
            },
            {
                "name": "8",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 256,
                    "c2": 256,
                    "n": 1,
                    "shortcut": True,
                },
                "from": ["7"],
            },
            {
                "name": "9",
                "module": "lovely_deep_learning.nn.block.SPPF",
                "args": {
                    "c1": 256,
                    "c2": 256,
                    "k": 5,
                },
                "from": ["8"],
            },
            {
                "name": "10",
                "module": "torch.nn.modules.upsampling.Upsample",
                "args": {
                    "size": None,
                    "scale_factor": 2,
                    "mode": "nearest",
                },
                "from": ["9"],
            },
            {
                "name": "11",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["10", "6"],
            },
            {
                "name": "12",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 384,
                    "c2": 128,
                    "n": 1,
                },
                "from": ["11"],
            },
            {
                "name": "13",
                "module": "torch.nn.modules.upsampling.Upsample",
                "args": {
                    "size": None,
                    "scale_factor": 2,
                    "mode": "nearest",
                },
                "from": ["12"],
            },
            {
                "name": "14",
                "module": "ultralytics.nn.modules.conv.Concat",
                "args": {
                    "dimension": 1,
                },
                "from": ["13", "4"],
            },
            {
                "name": "15",
                "module": "lovely_deep_learning.nn.block.C2f",
                "args": {
                    "c1": 192,
                    "c2": 64,
                    "n": 1,
                },
                "from": ["14"],
            },
            # {
            #     "name": "conv4",
            #     "module": "ultralytics.nn.modules.conv.Conv",
            #     "args": {
            #         "c1": 64,
            #         "c2": 64,
            #         "k": 3,
            #         "s": 2,
            #     },
            #     "from": ["input"],
            # },
            # {
            #     "name": "upsample",
            #     "module": "ultralytics.nn.modules.conv.Concat",
            #     "args": {
            #         "dimension": 1,
            #     },
            #     "from": ["conv2"],
            # },
            # {
            #     "name": "C2f_2",
            #     "module": "ultralytics.nn.modules.block.C2f",
            #     "args": {
            #         "c1": 192,
            #         "c2": 128,
            #         "n": 1,
            #     },
            #     "from": ["conv2"],
            # },
            # {
            #     "name": "conv4",
            #     "module": "ultralytics.nn.modules.conv.Conv",
            #     "args": {
            #         "c1": 128,
            #         "c2": 128,
            #         "k": 3,
            #         "s": 2,
            #     },
            #     "from": ["input"],
            # },
            # {
            #     "name": "upsample",
            #     "module": "ultralytics.nn.modules.conv.Concat",
            #     "args": {
            #         "dimension": 1,
            #     },
            #     "from": ["conv2"],
            # },
            # {
            #     "name": "C2f_2",
            #     "module": "ultralytics.nn.modules.block.C2f",
            #     "args": {
            #         "c1": 384,
            #         "c2": 256,
            #         "n": 1,
            #     },
            #     "from": ["conv2"],
            # },
            # {
            #     "name": "C2f_2",
            #     "module": "ultralytics.nn.modules.head.Detect",
            #     "args": {
            #         "nc": 80,
            #         "ch": [64,128,256],
            #         "n": 1,
            #     },
            #     "from": ["conv2"],
            # },
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
