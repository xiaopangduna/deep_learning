import torch
import torchvision.models as models
from ultralytics import YOLO

from lovely_deep_learning.model.DAGNet import DAGNet
from .utils import (
    efficientnet_v2_s_config,
    mobilenet_v3_large_config,
    regnet_y_32gf_config,
    resnet18_config,
    swin_v2_t_config,
    yolov8_n_config,
)


def test_DAGWeightLoader_resnet18():
    config = resnet18_config
    net = DAGNet(config["structure"])
    net.load_weights(**config["weight"])
    official_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    x = torch.randn(1, 3, 224, 224)
    official_out = official_resnet(x)
    dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGWeightLoader_efficientnet_v2_s():
    config = efficientnet_v2_s_config
    net = DAGNet(config["structure"])
    net.load_weights(**config["weight"])
    official = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    x = torch.randn(1, 3, 224, 224)
    official.eval()
    net.eval()
    with torch.no_grad():
        official_out = official(x)
        dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGWeightLoader_swin_v2_t():
    config = swin_v2_t_config
    net = DAGNet(config["structure"])
    net.load_weights(**config["weight"])
    official = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
    x = torch.randn(1, 3, 256, 256)
    official.eval()
    net.eval()
    with torch.no_grad():
        official_out = official(x)
        dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGWeightLoader_regnet_y_32gf():
    config = regnet_y_32gf_config
    net = DAGNet(config["structure"])
    net.load_weights(**config["weight"])
    official = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.DEFAULT)
    x = torch.randn(1, 3, 224, 224)
    official.eval()
    net.eval()
    with torch.no_grad():
        official_out = official(x)
        dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGWeightLoader_mobilenet_v3_large():
    config = mobilenet_v3_large_config
    net = DAGNet(config["structure"])
    net.load_weights(**config["weight"])
    official = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    x = torch.randn(1, 3, 224, 224)
    official.eval()
    net.eval()
    with torch.no_grad():
        official_out = official(x)
        dag_out = net([x])[0]
    assert torch.allclose(official_out, dag_out, atol=1e-6)


def test_DAGWeightLoader_yolov8_n():
    config = yolov8_n_config
    net = DAGNet(config["structure"])
    net.layers["22"].stride = torch.tensor([8, 16, 32], dtype=torch.float32)
    net.load_weights(**config["weight"])
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
