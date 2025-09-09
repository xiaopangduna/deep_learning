import torch
import torchvision.models.resnet
import re
import yaml
import torchvision.models as models
from lovely_deep_learning.models.DAGNet import DAGNet
from lovely_deep_learning.models.DAGWeightLoader import DAGWeightLoader
from .utils import resnet18_config
from .utils import load_yaml_config



def test_yaml_config_equal_dict_config():
    """测试函数：使用辅助函数读取YAML，通过assert对比内容"""
    #TODO 测试 yaml 配置文件是否与 dict 配置文件相等
    # configs/models/resnet18.yaml
    # 测试用例列表：(预期字典, YAML文件路径)
    test_cases = [
        (resnet18_config,"configs/models/resnet18.yaml"),
    ]
    
    for expected, yaml_path in test_cases:
        # 读取YAML（依赖辅助函数，出错直接抛出）
        actual = load_yaml_config(yaml_path)
        # 对比内容
        assert actual == expected, \
            f"配置不匹配 - 文件: {yaml_path}\n预期: {expected}\n实际: {actual}"


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
    config_dict = resnet18_config["structure"]
    config_yaml = load_yaml_config("configs/models/resnet18.yaml")
    loader_dict = 
    # loader_yaml = 
    assert config_dict == config_yaml["resnet18"]

    net = DAGNet(config_yaml["resnet18"])

    # 2. 加载官方 ResNet18 权重
    official_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    official_sd = official_resnet.state_dict()
    my_sd = net.state_dict()
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

    loader_config = {
        "path": "pretrained_models/resnet18-f37072fd.pth",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    }

    loader = DAGWeightLoader()
    loader.load_weights(net, **loader_config)
    # url = ("https://download.pytorch.org/models/resnet18-f37072fd.pth",)

    assert torch.allclose(official_stem_out, dag_stem_out, atol=1e-6)


def test_DAGWeightLoader():
    loader_config = {
        "class_path": "weight_loader.torchvision.TorchVisionWeightLoader",
        "init_args": {
            "path": "resnet50",
            "map_location": "cpu",
            "strict": False,
        },
    }

    loader = DAGWeightLoader()
    url = ("https://download.pytorch.org/models/resnet18-f37072fd.pth",)
    pass


