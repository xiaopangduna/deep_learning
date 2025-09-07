import torch
import torchvision.models.resnet
import torchvision.models as models
from lovely_deep_learning.models.DAGNet import DAGNet


def test_DAGNet_demo_model_two_inputs_one_outputs():
    config = {
        "inputs": [{"name": "img1", "shape": (3, 32, 32)}, {"name": "img2", "shape": (3, 32, 32)}],
        "outputs": [{"name": "class", "shape": (10,), "from": ["fc"]}],
        "layers": [
            {
                "name": "conv1",
                "module": "torch.nn.Conv2d",
                "args": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
                "from": ["img1"],
            },
            {
                "name": "conv2",
                "module": "torch.nn.Conv2d",
                "args": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
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
                "args": {"in_channels": 32, "out_channels": 32, "kernel_size": 3, "padding": 1},
                "from": ["concat"],
            },
            {"name": "flatten", "module": "torch.nn.Flatten", "args": {}, "from": ["conv3"]},
            {"name": "fc", "module": "torch.nn.LazyLinear", "args": {"out_features": 10}, "from": ["flatten"]},
        ],
    }
    #  {"name": "fc", "module": "torch.nn.Linear", "args": {"in_features":32*32*32, "out_features":10}, "from":["flatten"]}

    model = DAGNet(config)

    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(1, 3, 32, 32)

    out = model([x1, x2])

    assert out[0].shape == (1, 10)


# def test_DAGNet_equal_ResNet18():
#     resnet18_config = {
#         "inputs": [{"name": "input", "shape": (3, 224, 224)}],
#         "outputs": [{"name": "fc", "from": ["fc"]}],
#         "layers": [
#             # Stem
#             {
#                 "name": "conv1",
#                 "module": "torch.nn.Conv2d",
#                 "args": {
#                     "in_channels": 3,
#                     "out_channels": 64,
#                     "kernel_size": 7,
#                     "stride": 2,
#                     "padding": 3,
#                     "bias": False,
#                 },
#                 "from": ["input"],
#             },
#             {"name": "bn1", "module": "torch.nn.BatchNorm2d", "args": {"num_features": 64}, "from": ["conv1"]},
#             {"name": "relu", "module": "torch.nn.ReLU", "args": {"inplace": True}, "from": ["bn1"]},
#             {
#                 "name": "maxpool",
#                 "module": "torch.nn.MaxPool2d",
#                 "args": {"kernel_size": 3, "stride": 2, "padding": 1},
#                 "from": ["relu"],
#             },
#             # Layer1 (2x 64→64)
#             {
#                 "name": "layer1_block1",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
#                 "from": ["maxpool"],
#             },
#             {
#                 "name": "layer1_block2",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 64, "out_channels": 64, "stride": 1},
#                 "from": ["layer1_block1"],
#             },
#             # Layer2 (2x 64→128, stride=2)
#             {
#                 "name": "layer2_block1",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 64, "out_channels": 128, "stride": 2},
#                 "from": ["layer1_block2"],
#             },
#             {
#                 "name": "layer2_block2",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 128, "out_channels": 128, "stride": 1},
#                 "from": ["layer2_block1"],
#             },
#             # Layer3 (2x 128→256, stride=2)
#             {
#                 "name": "layer3_block1",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 128, "out_channels": 256, "stride": 2},
#                 "from": ["layer2_block2"],
#             },
#             {
#                 "name": "layer3_block2",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 256, "out_channels": 256, "stride": 1},
#                 "from": ["layer3_block1"],
#             },
#             # Layer4 (2x 256→512, stride=2)
#             {
#                 "name": "layer4_block1",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 256, "out_channels": 512, "stride": 2},
#                 "from": ["layer3_block2"],
#             },
#             {
#                 "name": "layer4_block2",
#                 "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
#                 "args": {"in_channels": 512, "out_channels": 512, "stride": 1},
#                 "from": ["layer4_block1"],
#             },
#             # Head
#             {
#                 "name": "avgpool",
#                 "module": "torch.nn.AdaptiveAvgPool2d",
#                 "args": {"output_size": (1, 1)},
#                 "from": ["layer4_block2"],
#             },
#             {"name": "flatten", "module": "torch.nn.Flatten", "args": {}, "from": ["avgpool"]},
#             {
#                 "name": "fc",
#                 "module": "torch.nn.Linear",
#                 "args": {"in_features": 512, "out_features": 1000},
#                 "from": ["flatten"],
#             },
#         ],
#     }

#     net = DAGNet(resnet18_config)

#     # 测试前向传播
#     x = torch.randn(1, 3, 224, 224)
#     out = net([x])

#     assert out[0].shape == (1, 1000)

def test_DAGNet_equal_ResNet18():
    # ------------------------
    # 1. DAGNet ResNet18 配置
    # ------------------------
    resnet18_config = {
        "inputs": [{"name": "input", "shape": (3, 224, 224)}],
        "outputs": [{"name": "fc", "from": ["fc"]}],
        "layers": [
            # Stem
            {"name": "conv1", "module": "torch.nn.Conv2d",
             "args": {"in_channels":3,"out_channels":64,"kernel_size":7,"stride":2,"padding":3,"bias":False},
             "from": ["input"]},
            {"name": "bn1", "module": "torch.nn.BatchNorm2d","args": {"num_features":64},"from": ["conv1"]},
            {"name": "relu", "module": "torch.nn.ReLU","args": {"inplace": True},"from": ["bn1"]},
            {"name": "maxpool", "module": "torch.nn.MaxPool2d","args": {"kernel_size":3,"stride":2,"padding":1},"from": ["relu"]},
            # Layer1
            {"name": "layer1_block1", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":64, "out_channels":64, "stride":1}, "from": ["maxpool"]},
            {"name": "layer1_block2", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":64, "out_channels":64, "stride":1}, "from": ["layer1_block1"]},
            # Layer2
            {"name": "layer2_block1", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":64, "out_channels":128, "stride":2}, "from": ["layer1_block2"]},
            {"name": "layer2_block2", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":128, "out_channels":128, "stride":1}, "from": ["layer2_block1"]},
            # Layer3
            {"name": "layer3_block1", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":128, "out_channels":256, "stride":2}, "from": ["layer2_block2"]},
            {"name": "layer3_block2", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":256, "out_channels":256, "stride":1}, "from": ["layer3_block1"]},
            # Layer4
            {"name": "layer4_block1", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":256, "out_channels":512, "stride":2}, "from": ["layer3_block2"]},
            {"name": "layer4_block2", "module": "lovely_deep_learning.nn.conv.DAGResidualBlock",
             "args": {"in_channels":512, "out_channels":512, "stride":1}, "from": ["layer4_block1"]},
            # Head
            {"name": "avgpool", "module": "torch.nn.AdaptiveAvgPool2d", "args": {"output_size": (1,1)}, "from": ["layer4_block2"]},
            {"name": "flatten", "module": "torch.nn.Flatten", "args": {}, "from": ["avgpool"]},
            {"name": "fc", "module": "torch.nn.Linear", "args": {"in_features":512, "out_features":1000}, "from": ["flatten"]},
        ]
    }

    net = DAGNet(resnet18_config)

    # ------------------------
    # 2. 加载官方 ResNet18 权重
    # ------------------------
    official_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    official_sd = official_resnet.state_dict()
    my_sd = net.state_dict()

    # ------------------------
    # 3. 权重映射表 (DAGResidualBlock -> BasicBlock)
    # ------------------------
    mapping = {}

    # Stem
    mapping.update({
        "conv1.weight": "conv1.weight",
        "bn1.weight": "bn1.weight",
        "bn1.bias": "bn1.bias",
        "bn1.running_mean": "bn1.running_mean",
        "bn1.running_var": "bn1.running_var",
    })

    # 辅助函数生成 Layer 映射
    def map_layer(layer_idx, block1_name, block2_name, in_c1, out_c1, in_c2, out_c2):
        layer_map = {}
        # Block1
        layer_map[f"layer{layer_idx}.0.conv1.weight"] = f"{block1_name}.conv1.weight"
        layer_map[f"layer{layer_idx}.0.bn1.weight"] = f"{block1_name}.bn1.weight"
        layer_map[f"layer{layer_idx}.0.bn1.bias"] = f"{block1_name}.bn1.bias"
        layer_map[f"layer{layer_idx}.0.bn1.running_mean"] = f"{block1_name}.bn1.running_mean"
        layer_map[f"layer{layer_idx}.0.bn1.running_var"] = f"{block1_name}.bn1.running_var"
        layer_map[f"layer{layer_idx}.0.conv2.weight"] = f"{block1_name}.conv2.weight"
        layer_map[f"layer{layer_idx}.0.bn2.weight"] = f"{block1_name}.bn2.weight"
        layer_map[f"layer{layer_idx}.0.bn2.bias"] = f"{block1_name}.bn2.bias"
        layer_map[f"layer{layer_idx}.0.bn2.running_mean"] = f"{block1_name}.bn2.running_mean"
        layer_map[f"layer{layer_idx}.0.bn2.running_var"] = f"{block1_name}.bn2.running_var"
        if in_c1 != out_c1:
            layer_map[f"layer{layer_idx}.0.downsample.0.weight"] = f"{block1_name}.downsample.0.weight"
            layer_map[f"layer{layer_idx}.0.downsample.1.weight"] = f"{block1_name}.downsample.1.weight"
            layer_map[f"layer{layer_idx}.0.downsample.1.bias"] = f"{block1_name}.downsample.1.bias"
            layer_map[f"layer{layer_idx}.0.downsample.1.running_mean"] = f"{block1_name}.downsample.1.running_mean"
            layer_map[f"layer{layer_idx}.0.downsample.1.running_var"] = f"{block1_name}.downsample.1.running_var"
        # Block2
        layer_map[f"layer{layer_idx}.1.conv1.weight"] = f"{block2_name}.conv1.weight"
        layer_map[f"layer{layer_idx}.1.bn1.weight"] = f"{block2_name}.bn1.weight"
        layer_map[f"layer{layer_idx}.1.bn1.bias"] = f"{block2_name}.bn1.bias"
        layer_map[f"layer{layer_idx}.1.bn1.running_mean"] = f"{block2_name}.bn1.running_mean"
        layer_map[f"layer{layer_idx}.1.bn1.running_var"] = f"{block2_name}.bn1.running_var"
        layer_map[f"layer{layer_idx}.1.conv2.weight"] = f"{block2_name}.conv2.weight"
        layer_map[f"layer{layer_idx}.1.bn2.weight"] = f"{block2_name}.bn2.weight"
        layer_map[f"layer{layer_idx}.1.bn2.bias"] = f"{block2_name}.bn2.bias"
        layer_map[f"layer{layer_idx}.1.bn2.running_mean"] = f"{block2_name}.bn2.running_mean"
        layer_map[f"layer{layer_idx}.1.bn2.running_var"] = f"{block2_name}.bn2.running_var"
        return layer_map

    # Layer1~4 映射
    mapping.update(map_layer(1, "layer1_block1", "layer1_block2", 64,64,64,64))
    mapping.update(map_layer(2, "layer2_block1", "layer2_block2", 64,128,128,128))
    mapping.update(map_layer(3, "layer3_block1", "layer3_block2", 128,256,256,256))
    mapping.update(map_layer(4, "layer4_block1", "layer4_block2", 256,512,512,512))

    # fc
    mapping["fc.weight"] = "fc.weight"
    mapping["fc.bias"] = "fc.bias"

    # ------------------------
    # 4. 加载权重
    # ------------------------
    new_sd = {}
    for k_off, k_my in mapping.items():
        if k_off in official_sd and k_my in my_sd:
            new_sd[k_my] = official_sd[k_off]
    net.load_state_dict(new_sd, strict=False)

    # ------------------------
    # 5. 前向验证
    # ------------------------
    x = torch.randn(1,3,224,224)
    out_dag = net([x])[0]
    out_official = official_resnet(x)

    assert out_dag.shape == out_official.shape
    assert torch.allclose(out_dag, out_official, atol=1e-6)

    print("DAGNet ResNet18 forward output matches official ResNet18!")


def test_all():
    from torchvision.models import resnet18

    # 1. 加载官方预训练模型
    model = resnet18(pretrained=True)

    # 2. 获取 state_dict
    state_dict = model.state_dict()

    # 3. 查看所有键名和形状
    for k, v in state_dict.items():
        print(k, v.shape)
