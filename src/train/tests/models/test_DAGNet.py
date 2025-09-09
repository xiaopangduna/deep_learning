import torch
import torchvision.models.resnet
import re
import torchvision.models as models
from lovely_deep_learning.models.DAGNet import DAGNet


def test_DAGNet_demo_model_two_inputs_one_outputs():
    config = {
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
    }
    #  {"name": "fc", "module": "torch.nn.Linear", "args": {"in_features":32*32*32, "out_features":10}, "from":["flatten"]}

    model = DAGNet(config)

    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(1, 3, 32, 32)

    out = model([x1, x2])

    assert out[0].shape == (1, 10)


def test_DAGNet_equal_ResNet18():
    # ------------------------
    # 1. DAGNet ResNet18 配置
    # ------------------------
    resnet18_config = {
        "inputs": [{"name": "input", "shape": (3, 224, 224)}],
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
            {"name": "avgpool", "module": "torch.nn.AdaptiveAvgPool2d", "args": {"output_size": (1,1)}, "from": ["layer4"]},
            {"name": "flatten", "module": "torch.nn.Flatten", "args": {}, "from": ["avgpool"]},
            {"name": "fc", "module": "torch.nn.Linear", "args": {"in_features":512, "out_features":1000}, "from": ["flatten"]},
        ],
    }

    net = DAGNet(resnet18_config)

    # ------------------------
    # 2. 加载官方 ResNet18 权重
    # ------------------------
    official_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    official_sd = official_resnet.state_dict()
    my_sd = net.state_dict()

    # new_sd = {mapping_fn(k): v for k, v in official_sd.items()}
    new_sd = {"layers." + k: v for k, v in official_sd.items()}
    compatible_sd = {k: v for k, v in new_sd.items() if k in my_sd}
    my_sd.update(compatible_sd)
    net.load_state_dict(my_sd)

    x = torch.randn(1, 3, 224, 224)
    net.eval()
    official_resnet.eval()
    # with torch.no_grad():
    #     # 官方 ResNet18 stem: conv1->bn1->relu->maxpool
    #     official_stem_out = official_resnet.conv1(x)
    #     official_stem_out = official_resnet.bn1(official_stem_out)
    #     official_stem_out = official_resnet.relu(official_stem_out)
    #     official_stem_out = official_resnet.maxpool(official_stem_out)
    #     official_stem_out = official_resnet.layer1(official_stem_out)
    #     official_stem_out = official_resnet.layer2(official_stem_out)
    #     official_stem_out = official_resnet.layer3(official_stem_out)
    #     official_stem_out = official_resnet.layer4(official_stem_out)
    #     official_stem_out = official_resnet.avgpool(official_stem_out)
    official_stem_out = official_resnet(x)
    # DAGNet forward 输出对应 stem
    dag_stem_out = net([x])[0]  # 输出 fc 层对应的 from="relu" 或 maxpool

    assert torch.allclose(official_stem_out, dag_stem_out, atol=1e-6)

