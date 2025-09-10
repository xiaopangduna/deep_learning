import torch
# import torchvision.models.resnet
# import re
# import yaml
import torchvision.models as models

from lovely_deep_learning.models.DAGNet import DAGNet
from lovely_deep_learning.models.DAGWeightLoader import DAGWeightLoader
from .utils import *


def test_yaml_config_equal_dict_config():
    """测试函数：使用辅助函数读取YAML，通过assert对比内容"""
    test_cases = [
        (resnet18_config, "configs/models/resnet18.yaml"),
    ]

    for expected, yaml_path in test_cases:
        actual = load_yaml_config(yaml_path)
        assert actual == expected, f"配置不匹配 - 文件: {yaml_path}\n预期: {expected}\n实际: {actual}"


def test_DAGNet_demo_model_two_inputs_one_outputs():
    config = demo_config_two_inputs_one_outputs
    model = DAGNet(config["structure"])

    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(1, 3, 32, 32)

    out = model([x1, x2])

    assert out[0].shape == (1, 10)


def test_DAGNet_equal_ResNet18():
    config = resnet18_config

    net = DAGNet(config["structure"])

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
    dag_stem_out = net([x])[0]  # 输出 fc 层对应的 from="relu" 或 maxpool

    # loader = DAGWeightLoader()
    # loader.load_weights(net, **loader_config)
    # url = ("https://download.pytorch.org/models/resnet18-f37072fd.pth",)

    assert torch.allclose(official_stem_out, dag_stem_out, atol=1e-6)


def test_DAGWeightLoader_resnet18():
    config = resnet18_config
    net = DAGNet(config["structure"])
    loader = DAGWeightLoader()
    loader.load_weights(net, **config["weight"])
    official_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    x = torch.randn(1, 3, 224, 224)
    official_out = official_resnet(x)
    dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGNet_equal_yolov8():

    
    pass
