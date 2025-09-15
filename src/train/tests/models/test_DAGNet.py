import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

from lovely_deep_learning.models.DAGNet import DAGNet
from lovely_deep_learning.models.DAGWeightLoader import DAGWeightLoader
from .utils import *


def test_yaml_config_equal_dict_config():
    """测试函数：使用辅助函数读取YAML，通过assert对比内容"""
    test_cases = [
        (resnet18_config, "configs/models/resnet18.yaml"),
        (yolov8_n_config, "configs/models/yolov8_n.yaml"),
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

    net.eval()
    dag_stem_out = net([x])[0]  # 输出 fc 层对应的 from="relu" 或 maxpool

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


def test_DAGNet_equal_yolov8_n():
    torch.manual_seed(42)
    official_model = YOLO(r"pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
    config = yolov8_n_config
    net = DAGNet(config["structure"])

    official_sd = official_model.state_dict()
    my_sd = net.state_dict()
    new_sd = {"layers." + k[12:]: v for k, v in official_sd.items()}
    compatible_sd = {k: v for k, v in new_sd.items() if k in my_sd}
    my_sd.update(compatible_sd)
    net.load_state_dict(my_sd)

    x = torch.randn(1, 3, 640, 640)

    official_model.eval()
    official_out_last = official_model.model(x)
    with torch.no_grad():
        official_out = official_model.model.model[0](x)
        official_out = official_model.model.model[1](official_out)
        official_out = official_model.model.model[2](official_out)
        official_out = official_model.model.model[3](official_out)
        official_out = official_model.model.model[4](official_out)
        official_out_4 = official_out.clone()
        official_out = official_model.model.model[5](official_out)
        official_out = official_model.model.model[6](official_out)
        official_out_6 = official_out.clone()
        official_out = official_model.model.model[7](official_out)
        official_out = official_model.model.model[8](official_out)
        official_out = official_model.model.model[9](official_out)
        official_out_9 = official_out.clone()
        official_out = official_model.model.model[10](official_out)
        official_out = official_model.model.model[11]([official_out, official_out_6])
        official_out = official_model.model.model[12](official_out)
        official_out_12 = official_out.clone()
        official_out = official_model.model.model[13](official_out)
        official_out = official_model.model.model[14]([official_out, official_out_4])
        official_out = official_model.model.model[15](official_out)
        official_out_15 = official_out.clone()
        official_out = official_model.model.model[16](official_out)
        official_out = official_model.model.model[17]([official_out, official_out_12])
        official_out = official_model.model.model[18](official_out)
        official_out_18 = official_out.clone()
        official_out = official_model.model.model[19](official_out)
        official_out = official_model.model.model[20]([official_out, official_out_9])
        official_out = official_model.model.model[21](official_out)
        official_out_21 = official_out.clone()
        official_out = official_model.model.model[22]([official_out_15, official_out_18, official_out_21])

    net.eval()
    dag_out = net([x])[0]

    print(official_model.model.model[22])
    print(net.layers["22"])

    # assert torch.allclose(official_out, dag_out, atol=1e-6)

    # assert torch.allclose(official_out[0], official_out_last[0], atol=1e-6)
    # assert torch.allclose(official_out[1][0], official_out_last[1][0], atol=1e-6)
    # assert torch.allclose(official_out[1][1], official_out_last[1][1], atol=1e-6)
    # assert torch.allclose(official_out[1][2], official_out_last[1][2], atol=1e-6)

    assert torch.allclose(dag_out[1][0], official_out_last[1][0], atol=1e-6)
    assert torch.allclose(dag_out[1][1], official_out_last[1][1], atol=1e-6)
    assert torch.allclose(dag_out[1][2], official_out_last[1][2], atol=1e-6)

    assert torch.allclose(official_out[0], dag_out[0], atol=1e-6)


def test_DAGWeightLoader_yolov8_n():
    config = yolov8_n_config
    net = DAGNet(config["structure"])
    net.layers["22"].stride =  torch.tensor([8,16,32], dtype=torch.float32)
    DAGWeightLoader().load_weights(net, **config["weight"]) 
    official_model = YOLO(r"pretrained_models/yolov8n.pt")  
    official_model.eval()
    net.eval()
    x = torch.randn(1, 3, 640, 640)
    official_out = official_model.model(x)
    dag_out = net([x])[0]

    assert torch.allclose(dag_out[1][0], official_out[1][0], atol=1e-6)
    assert torch.allclose(dag_out[1][1], official_out[1][1], atol=1e-6)
    assert torch.allclose(dag_out[1][2], official_out[1][2], atol=1e-6)

    assert torch.allclose(official_out[0], dag_out[0], atol=1e-4)
